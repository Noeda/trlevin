use crate::numbers::ContextModelNumber;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::mem::drop;

pub use crate::numbers::Enumerable;
pub use ordered_float::OrderedFloat;

/// Implement this trait for your problem so that the search and optimization knows how to interact
/// with it.
pub trait ContextModelable {
    type Action: Enumerable;
    type Context: Enumerable;
    type State;

    /// Return true if some state is considered a solution.
    fn is_solution(&self, state: &Self::State) -> bool;

    /// Given environment and state, fill the given vector with possible actions.
    ///
    /// The vector is always passed as an empty vector.
    fn possible_actions(&self, state: &Self::State, actions: &mut Vec<Self::Action>);

    /// Applies an action on an environment and state. There is an assumption that possible_actions
    /// has been consulted before calling this.
    ///
    /// Put the new state in new_state.
    fn perform_action(
        &self,
        state: &Self::State,
        action: &Self::Action,
        new_state: &mut Self::State,
    );

    /// Given an environment and state, fill a vector of integers that identifh the currently
    /// active contexes. The vector is always passed as an empty vector.
    fn active_contexts(&self, state: &Self::State, active_contexts: &mut Vec<Self::Context>);
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ContextModelParameters<T> {
    // a vector of context mutexes,
    // which are vectors of possible contexts in each mutex,
    // which contain parameter vector for that context.
    context_parameters: Vec<Vec<Vec<T>>>,
    e_mix: T,
    e_low: T,
}

impl<T: ContextModelNumber> ContextModelParameters<T> {
    pub fn new(context_mutex_sizes: &[usize], nactions: usize) -> Self {
        let mut one_divided_by_nactions: T = T::one();
        one_divided_by_nactions.inplace_div(&T::from_usize(nactions));

        let mut context_parameters = Vec::with_capacity(context_mutex_sizes.len());
        for &size in context_mutex_sizes {
            let mut mutex = Vec::with_capacity(size);
            for _ in 0..size {
                let mut context = Vec::with_capacity(nactions);
                for _ in 0..nactions {
                    context.push(one_divided_by_nactions.clone());
                }
                mutex.push(context);
            }
            context_parameters.push(mutex);
        }
        ContextModelParameters {
            context_parameters,
            e_mix: T::from_f64(0.001),
            e_low: T::from_f64(0.0001),
        }
    }

    pub fn num_parameters(&self) -> usize {
        let mut result: usize = 0;
        for mutex in self.context_parameters.iter() {
            for context in mutex.iter() {
                result += context.len();
            }
        }
        result
    }

    pub fn num_actions(&self) -> usize {
        if self.context_parameters.is_empty() || self.context_parameters[0].is_empty() {
            0
        } else {
            self.context_parameters[0][0].len()
        }
    }

    pub fn new_gradient_state(&self) -> ContextModelParametersGradientState<T> {
        ContextModelParametersGradientState { g_tmps: vec![] }
    }

    pub fn set_zero(&mut self) {
        for vec1 in self.context_parameters.iter_mut() {
            for vec2 in vec1.iter_mut() {
                for val in vec2.iter_mut() {
                    *val = T::zero();
                }
            }
        }
    }

    pub fn e_mix(&self) -> T {
        self.e_mix.clone()
    }

    pub fn set_e_mix(&mut self, e_mix: T) {
        self.e_mix = e_mix;
    }
}

// Item used as work items in the search
// Has non-standard Ord, Eq and friends.
#[derive(Clone)]
struct WorkItem<T, Action> {
    d_p: T, // d divided by pi
    depth: usize,
    pi: T,
    state_idx: usize,
    action: Option<Action>,
}

// Custom Ord to make WorkItems work as a min-heap in BinaryHeap
// Also, ordering only depends on d_p.
impl<T: PartialOrd, Action> PartialOrd for WorkItem<T, Action> {
    fn partial_cmp(&self, other: &WorkItem<T, Action>) -> Option<Ordering> {
        other.d_p.partial_cmp(&self.d_p)
    }
}

impl<T: Ord, Action> Ord for WorkItem<T, Action> {
    fn cmp(&self, other: &WorkItem<T, Action>) -> Ordering {
        other.d_p.cmp(&self.d_p)
    }
}

impl<T: Eq, Action> Eq for WorkItem<T, Action> {}

impl<T: PartialEq, Action> PartialEq for WorkItem<T, Action> {
    fn eq(&self, other: &WorkItem<T, Action>) -> bool {
        self.d_p == other.d_p
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum LevinSearchSolution<A, B> {
    Path(Vec<(A, B)>, A),
}

/// Given a Levin Tree Search model, executes a search with given budget.
///
/// Returns either the found solution, or a handle that can be used to continue the search.
///
/// The returned path is composed of (state, action) pairs, plus the final state. The state is *before* the action has
/// been done, so it includes the initial state.
pub fn levin_search<T: ContextModelNumber, Env>(
    env: &Env,
    initial_state: Env::State,
    context_parameters: &ContextModelParameters<T>,
    budget: Option<usize>,
) -> Option<LevinSearchSolution<Env::State, Env::Action>>
where
    Env: ContextModelable,
    Env::State: Clone + Ord,
    Env::Action: Clone,
{
    assert!(!context_parameters.context_parameters.is_empty());
    assert!(!context_parameters.context_parameters[0].is_empty());

    let mut work_queue: BinaryHeap<WorkItem<T, Env::Action>> = BinaryHeap::new();
    // FIXME: Can we solve having two copies of states in node_states and visited_states. Have
    // visited_states point to node_states. Or have some structure that is both associative table
    // and can be used like vector.
    #[allow(clippy::type_complexity)]
    let mut node_states: Vec<(Env::State, Option<Env::Action>, Option<usize>)> = vec![];
    let mut visited_states: BTreeMap<Env::State, T> = BTreeMap::new();

    node_states.push((initial_state, None, None));

    let mut active_contexts: Vec<usize> = vec![];
    let mut active_contexts_high_level: Vec<Env::Context> = vec![];
    let mut possible_actions: Vec<Env::Action> = vec![];

    let nactions: usize = context_parameters.context_parameters[0][0].len();
    let nctx_mutexes: usize = context_parameters.context_parameters.len();

    let mut expansions: usize = 0;

    let mut pcs: Vec<T> = vec![];

    let mut one_minus_emix: T = T::one();
    one_minus_emix.inplace_sub(&context_parameters.e_mix);

    work_queue.push(WorkItem {
        d_p: T::zero(),
        depth: 0,
        pi: T::one(),
        state_idx: 0,
        action: None,
    });

    loop {
        // Pop from work queue, if work queue is empty, then no solution
        let work_item = match work_queue.pop() {
            Some(work_item) => work_item,
            None => return None,
        };

        // Is this the solution?
        if env.is_solution(&node_states[work_item.state_idx].0) {
            let mut result_path: Vec<(Env::State, Env::Action)> = vec![];
            let last_state: Env::State = node_states[work_item.state_idx].0.clone();
            if work_item.action.is_none() {
                return Some(LevinSearchSolution::Path(result_path, last_state));
            }
            let mut backtrack_state_idx: usize = work_item.state_idx;
            loop {
                if node_states[backtrack_state_idx].1.is_none() {
                    break;
                }
                if let Some(prev_idx) = node_states[backtrack_state_idx].2 {
                    let tmp = backtrack_state_idx;
                    backtrack_state_idx = prev_idx;
                    result_path.push((
                        node_states[backtrack_state_idx].0.clone(),
                        node_states[tmp].1.clone().unwrap(),
                    ));
                    continue;
                }
                unreachable!();
            }
            result_path.reverse();
            return Some(LevinSearchSolution::Path(result_path, last_state));
        }

        let mut new_state: Env::State = node_states[work_item.state_idx].0.clone();

        // The work item has an action that we execute to get a new state. (Except for very first
        // node where action is None)
        if let Some(ref action) = work_item.action {
            env.perform_action(&node_states[work_item.state_idx].0, action, &mut new_state);
        }

        // Have we been in this state before?
        let mut state_pi: T = T::zero();
        if let Some(old_state_pi) = visited_states.get(&new_state) {
            state_pi = old_state_pi.clone();
        }

        // Search pruning
        if state_pi >= work_item.pi {
            continue;
        }
        drop(state_pi); // make sure we don't accidentally use this later

        expansions += 1;
        if let Some(budget) = budget {
            if expansions == budget {
                return None;
            }
        }

        let mut node: usize = node_states.len();
        if work_item.action.is_some() {
            node_states.push((
                new_state,
                work_item.action.clone(),
                Some(work_item.state_idx),
            ));
        } else {
            node = 0;
        }
        let new_state: &Env::State = &node_states.last().unwrap().0;

        // What contexts are active?
        active_contexts_high_level.truncate(0);
        env.active_contexts(new_state, &mut active_contexts_high_level);
        active_contexts.truncate(0);
        for ctx in active_contexts_high_level.drain(0..) {
            active_contexts.push(ctx.to_number());
        }
        assert_eq!(active_contexts.len(), nctx_mutexes);

        // And which actions can be done?
        possible_actions.truncate(0);
        env.possible_actions(new_state, &mut possible_actions);

        pcs.truncate(0);
        for act in possible_actions.iter() {
            for (ctx_mutex_idx, ctx_id) in active_contexts.iter().enumerate() {
                pcs.push(pc(
                    env,
                    &context_parameters.context_parameters[ctx_mutex_idx],
                    act,
                    ctx_id,
                ));
            }
        }

        let normalizer: T = compute_normalizer(env, &possible_actions, &pcs, &active_contexts);
        for (act_idx, act) in possible_actions.drain(0..).enumerate() {
            let mut product_mix: T = T::one();
            for (ctx_mutex_idx, _ctx_id) in active_contexts.iter().enumerate() {
                product_mix.inplace_mul(&pcs[ctx_mutex_idx + act_idx * active_contexts.len()]);
            }
            product_mix.inplace_mul(&normalizer);

            // pi * ((1 - e_mix) * product_mix + e_mix / cardinality(actions))
            let mut action_probability: T = work_item.pi.clone();
            let mut e_mix_div_card = context_parameters.e_mix.clone();
            e_mix_div_card.inplace_div(&T::from_usize(nactions));
            product_mix.inplace_mul(&one_minus_emix);
            product_mix.inplace_add(&e_mix_div_card);
            action_probability.inplace_mul(&product_mix);

            let depth_plus_one: usize = work_item.depth + 1;
            let mut depth_plus_one: T = T::from_usize(depth_plus_one);
            depth_plus_one.inplace_div(&action_probability);

            work_queue.push(WorkItem {
                d_p: depth_plus_one,
                depth: work_item.depth + 1,
                pi: action_probability,
                state_idx: node,
                action: Some(act),
            });
        }
        visited_states.insert(new_state.clone(), work_item.pi.clone());
    }
}

// TODO: simplify the signature here, env and acts don't need to be passed like that.
fn compute_normalizer<T: ContextModelNumber, Env: ContextModelable>(
    _env: &Env,
    acts: &[Env::Action],
    pcs: &[T],
    active_contexts: &[usize],
) -> T {
    let mut accum = T::zero();
    for (act_idx, _act) in acts.iter().enumerate() {
        let mut mul_accum = T::one();
        for (ctx_mutex_idx, _ctx_idx) in active_contexts.iter().enumerate() {
            let pc_value = &pcs[ctx_mutex_idx + act_idx * active_contexts.len()];
            mul_accum.inplace_mul(pc_value);
        }
        accum.inplace_add(&mul_accum);
    }
    let mut one = T::one();
    one.inplace_div(&accum);
    one
}

fn pc<T: ContextModelNumber, Env: ContextModelable>(
    _env: &Env,
    context_parameters: &[Vec<T>],
    action: &Env::Action,
    ctx: &usize,
) -> T {
    let nactions: usize = context_parameters[*ctx].len();
    let action_num: usize = action.to_number();
    assert!(
        action_num < nactions,
        "Invalid action number, out of range."
    );

    let mut numerator: T = context_parameters[*ctx][action_num].clone();
    let mut max_v: T = numerator.clone();

    for val in context_parameters[*ctx].iter() {
        if val > &max_v {
            max_v = val.clone();
        }
    }
    numerator.inplace_sub(&max_v);
    numerator = numerator.exp();

    let mut denominator: T = T::zero();
    for val in context_parameters[*ctx].iter() {
        let mut val = val.clone();
        val.inplace_sub(&max_v);
        val = val.exp();
        denominator.inplace_add(&val);
    }

    numerator.inplace_div(&denominator);
    numerator
}

pub fn levin_loss<T, Env>(
    context_parameters: &ContextModelParameters<T>,
    env: &Env,
    path: &[(Env::State, Env::Action)],
) -> T
where
    T: ContextModelNumber,
    Env: ContextModelable,
{
    // initialize with depth
    let mut result: T = T::from_usize(path.len());

    let mut possible_actions = Vec::new();
    let mut ctxs: Vec<Env::Context> = Vec::new();

    let c: &[Vec<Vec<T>>] = &context_parameters.context_parameters;

    for (state, action) in path.iter() {
        possible_actions.truncate(0);
        ctxs.truncate(0);
        env.possible_actions(state, &mut possible_actions);
        env.active_contexts(state, &mut ctxs);
        let mut result2: T = T::zero();
        for act in possible_actions.drain(0..) {
            let mut result3: T = T::zero();
            for (ctx_mutex_idx, ctx) in ctxs.iter().enumerate() {
                let ctx = ctx.to_number();
                // accum = accum + B(ctx, possible action) - B(ctx, taken action)
                let possible_action_param: &T = &c[ctx_mutex_idx][ctx][act.to_number()];
                let next_action_param: &T = &c[ctx_mutex_idx][ctx][action.to_number()];
                result3.inplace_add(possible_action_param);
                result3.inplace_sub(next_action_param);
            }
            result3 = result3.exp();
            result2.inplace_add(&result3);
        }
        result.inplace_mul(&result2);
    }

    // compute regularization term

    // b0 = (1 - 1/num_actions) * log epsilon_low
    let mut b0_1 = T::from_usize(context_parameters.num_actions());
    b0_1 = b0_1.recip();
    let mut b0 = T::one();
    b0.inplace_sub(&b0_1);
    let e_low = context_parameters.e_low.clone().ln();
    b0.inplace_mul(&e_low);

    let mut regularizer: T = T::zero();
    for vec1 in c.iter() {
        for vec2 in vec1.iter() {
            for val in vec2.iter() {
                let mut v = val.clone();
                v.inplace_sub(&b0);
                let v2 = v.clone();
                v.inplace_mul(&v2);
                regularizer.inplace_add(&v);
            }
        }
    }
    regularizer = regularizer.sqrt();
    regularizer.inplace_mul(&T::from_usize(5));

    result.inplace_add(&regularizer);

    result
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ContextModelParametersGradientState<T: ContextModelNumber> {
    g_tmps: Vec<(T, Vec<Vec<Vec<T>>>)>,
}

/// Computes the gradient of context parameters that, if descended upon, minimizes levin_loss()
///
/// The 'gradient' is not zeroed before call. This is to make it possible to accumulate gradients.
/// Use gradient.set_zero() before calling if you want to reset the gradient.
///
/// The gradient_state contains intermediate data used during gradient computation. Pass it in to
/// avoid allocating memory. You can make it with .new_gradient_state() in ContextModelParameters.
pub fn loss_gradient<T, Env>(
    context_parameters: &ContextModelParameters<T>,
    gradient: &mut ContextModelParameters<T>,
    gradient_state: &mut ContextModelParametersGradientState<T>,
    env: &Env,
    path: &[(Env::State, Env::Action)],
) where
    T: ContextModelNumber,
    Env: ContextModelable,
{
    let g: &mut [Vec<Vec<T>>] = &mut gradient.context_parameters;
    let c: &[Vec<Vec<T>>] = &context_parameters.context_parameters;

    fn alloc_zeros<T: Clone>(shape: &[Vec<Vec<T>>], item: T) -> Vec<Vec<Vec<T>>> {
        let mut result = Vec::with_capacity(shape.len());
        for vec1 in shape.iter() {
            let mut vec1_result = Vec::with_capacity(vec1.len());
            for vec2 in vec1.iter() {
                let mut vec2_result = Vec::with_capacity(vec2.len());
                for _ in vec2.iter() {
                    vec2_result.push(item.clone());
                }
                vec1_result.push(vec2_result);
            }
            result.push(vec1_result);
        }
        result
    }

    let mut result: T = T::from_usize(path.len());

    let mut possible_actions = Vec::new();
    let mut ctxs = Vec::new();

    for (path_idx, (state, action)) in path.iter().enumerate() {
        possible_actions.truncate(0);
        ctxs.truncate(0);
        env.possible_actions(state, &mut possible_actions);
        env.active_contexts(state, &mut ctxs);

        if gradient_state.g_tmps.len() <= path_idx {
            let g_tmp: Vec<Vec<Vec<T>>> = alloc_zeros(c, T::zero());
            gradient_state.g_tmps.push((T::zero(), g_tmp));
        }
        let g_tmp: &mut [Vec<Vec<T>>] = &mut gradient_state.g_tmps[path_idx].1;
        for v in g_tmp.iter_mut() {
            for v2 in v.iter_mut() {
                for v3 in v2.iter_mut() {
                    *v3 = T::zero();
                }
            }
        }

        let mut result2: T = T::zero();
        for act in possible_actions.drain(0..) {
            // use g_tmps[idx+1] as temporary area for g_tmp2
            if gradient_state.g_tmps.len() <= path_idx + 1 {
                let g_tmp: Vec<Vec<Vec<T>>> = alloc_zeros(c, T::zero());
                gradient_state.g_tmps.push((T::zero(), g_tmp));
            }
            let g_tmp2 = &mut gradient_state.g_tmps[path_idx + 1].1;
            for v in g_tmp2.iter_mut() {
                for v2 in v.iter_mut() {
                    for v3 in v2.iter_mut() {
                        *v3 = T::zero();
                    }
                }
            }

            let mut result3: T = T::zero();
            for (ctx_mutex_idx, ctx) in ctxs.iter().enumerate() {
                let ctx = ctx.to_number();
                // accum = accum + B(ctx, possible action) - B(ctx, taken action)
                let possible_action_param: &T = &c[ctx_mutex_idx][ctx][act.to_number()];
                let next_action_param: &T = &c[ctx_mutex_idx][ctx][action.to_number()];
                result3.inplace_add(possible_action_param);
                result3.inplace_sub(next_action_param);

                g_tmp2[ctx_mutex_idx][ctx][act.to_number()].inplace_add(&T::one());
                g_tmp2[ctx_mutex_idx][ctx][action.to_number()].inplace_sub(&T::one());
            }
            result3 = result3.exp();
            result2.inplace_add(&result3);

            for ctx_mutex_idx in 0..ctxs.len() {
                for ctx_id in 0..gradient_state.g_tmps[path_idx].1[ctx_mutex_idx].len() {
                    for act_idx in 0..gradient_state.g_tmps[path_idx].1[ctx_mutex_idx][ctx_id].len()
                    {
                        gradient_state.g_tmps[path_idx + 1].1[ctx_mutex_idx][ctx_id][act_idx]
                            .inplace_mul(&result3);
                        let v = gradient_state.g_tmps[path_idx + 1].1[ctx_mutex_idx][ctx_id]
                            [act_idx]
                            .clone();
                        gradient_state.g_tmps[path_idx].1[ctx_mutex_idx][ctx_id][act_idx]
                            .inplace_add(&v);
                    }
                }
            }
        }
        result.inplace_mul(&result2);
        gradient_state.g_tmps[path_idx].0 = result2;
    }

    for (mul, g_tmp) in gradient_state.g_tmps.iter().take(path.len()) {
        for (idx1, vec) in g_tmp.iter().enumerate() {
            for (idx2, vec2) in vec.iter().enumerate() {
                for (idx3, param) in vec2.iter().enumerate() {
                    let mut v: T = param.clone();
                    v.inplace_mul(&result);
                    v.inplace_div(mul);
                    g[idx1][idx2][idx3].inplace_add(&v);
                }
            }
        }
    }

    // regularizer
    let mut b0_1 = T::from_usize(context_parameters.num_actions());
    b0_1 = b0_1.recip();
    let mut b0 = T::one();
    b0.inplace_sub(&b0_1);
    let e_low = context_parameters.e_low.clone().ln();
    b0.inplace_mul(&e_low);

    if gradient_state.g_tmps.is_empty() {
        let g_tmp: Vec<Vec<Vec<T>>> = alloc_zeros(c, T::zero());
        gradient_state.g_tmps.push((T::zero(), g_tmp));
    }
    let g_tmp: &mut [Vec<Vec<T>>] = &mut gradient_state.g_tmps[0].1;

    let mut regularizer: T = T::zero();
    for (idx, vec1) in c.iter().enumerate() {
        for (idx2, vec2) in vec1.iter().enumerate() {
            for (idx3, val) in vec2.iter().enumerate() {
                let mut v = val.clone();
                v.inplace_sub(&b0);

                let mut gv = v.clone();
                gv.inplace_mul(&T::from_usize(2));
                // derivative:
                // (v - b0)**2  ->  2 * (v - b0)
                g_tmp[idx][idx2][idx3].inplace_add(&gv);

                let v2 = v.clone();
                v.inplace_mul(&v2);
                regularizer.inplace_add(&v);
            }
        }
    }
    regularizer = regularizer.sqrt();
    for (idx, vec1) in c.iter().enumerate() {
        for (idx2, vec2) in vec1.iter().enumerate() {
            for (idx3, _val) in vec2.iter().enumerate() {
                let mut v = regularizer.clone();
                v.inplace_mul(&T::from_usize(2));
                v = v.recip();
                g_tmp[idx][idx2][idx3].inplace_mul(&v);
                g_tmp[idx][idx2][idx3].inplace_mul(&T::from_usize(5));
                g[idx][idx2][idx3].inplace_add(&g_tmp[idx][idx2][idx3]);
            }
        }
    }
}

struct LineSearchConfig<'a, T, Env: ContextModelable> {
    initial_scalar: T,
    env: &'a Env,
    paths: &'a [Vec<(Env::State, Env::Action)>],
}

/// Configures how to apply gradients.
pub struct GradientConfig<'a, T, Env: ContextModelable> {
    gradient_clip: Option<T>,
    // If true, then will try varying step sizes to minimize loss along the gradient.
    // The value is the first scalar to try. If line search is used, then the environment and all
    // paths being used must be supplied as well because line search needs to evaluate loss at
    // various points.
    do_line_search: Option<LineSearchConfig<'a, T, Env>>,
    // If loss changes only by this much, consider a step done.
    stop_at_loss_change: T,
}

impl<'a, T: ContextModelNumber, Env: ContextModelable> GradientConfig<'a, T, Env> {
    /// The default clips gradients to (-0.1, 0.1)
    pub fn new() -> Self {
        GradientConfig {
            gradient_clip: Some(T::from_f64(0.1)),
            do_line_search: None,
            stop_at_loss_change: T::from_f64(0.0001),
        }
    }

    pub fn no_gradient_clip(mut self) -> Self {
        self.gradient_clip = None;
        self
    }

    pub fn gradient_clip(mut self, gradient_clip: T) -> Self {
        self.gradient_clip = Some(gradient_clip);
        self
    }

    pub fn do_line_search(
        mut self,
        do_line_search: T,
        env: &'a Env,
        paths: &'a [Vec<(Env::State, Env::Action)>],
    ) -> Self {
        self.do_line_search = Some(LineSearchConfig {
            initial_scalar: do_line_search,
            env,
            paths,
        });
        self
    }
}

impl<'a, T: ContextModelNumber, Env: ContextModelable> Default for GradientConfig<'a, T, Env> {
    fn default() -> Self {
        GradientConfig::new()
    }
}

/// Applies a gradient to parameters.
pub fn apply_gradient<T: ContextModelNumber, Env: ContextModelable>(
    context_parameters: &mut ContextModelParameters<T>,
    gradient: &ContextModelParameters<T>,
    gradient_config: &GradientConfig<T, Env>,
) {
    if let Some(ref conf) = gradient_config.do_line_search {
        apply_gradient_line_search(
            context_parameters,
            conf.env,
            conf.paths,
            gradient,
            gradient_config,
            &conf.initial_scalar,
        );
    } else {
        for idx in 0..context_parameters.context_parameters.len() {
            for idx2 in 0..context_parameters.context_parameters[idx].len() {
                for idx3 in 0..context_parameters.context_parameters[idx][idx2].len() {
                    let mut v = gradient.context_parameters[idx][idx2][idx3].clone();
                    if let Some(ref gradient_clip) = gradient_config.gradient_clip {
                        let mut neg_gradient_clip = gradient_clip.clone();
                        neg_gradient_clip.inplace_mul(&T::from_f64(-1.0));
                        if &v > gradient_clip {
                            v = gradient_clip.clone();
                        } else if v < neg_gradient_clip {
                            v = neg_gradient_clip;
                        }
                    }
                    context_parameters.context_parameters[idx][idx2][idx3].inplace_sub(&v);
                }
            }
        }
    }
}

fn apply_gradient_line_search<T, Env>(
    context_parameters: &mut ContextModelParameters<T>,
    env: &Env,
    paths: &[Vec<(Env::State, Env::Action)>],
    gradient: &ContextModelParameters<T>,
    gradient_config: &GradientConfig<T, Env>,
    initial_step_size: &T,
) where
    T: ContextModelNumber,
    Env: ContextModelable,
{
    let mut scale: T = initial_step_size.clone();
    let mut base_loss: T = T::zero();
    for solution in paths.iter() {
        base_loss.inplace_add(&levin_loss(context_parameters, env, solution));
    }
    loop {
        for idx in 0..context_parameters.context_parameters.len() {
            for idx2 in 0..context_parameters.context_parameters[idx].len() {
                for idx3 in 0..context_parameters.context_parameters[idx][idx2].len() {
                    let mut v = gradient.context_parameters[idx][idx2][idx3].clone();
                    if let Some(ref gradient_clip) = gradient_config.gradient_clip {
                        let mut neg_gradient_clip = gradient_clip.clone();
                        neg_gradient_clip.inplace_mul(&T::from_f64(-1.0));
                        if &v > gradient_clip {
                            v = gradient_clip.clone();
                        } else if v < neg_gradient_clip {
                            v = neg_gradient_clip;
                        }
                    }
                    v.inplace_mul(&scale);
                    context_parameters.context_parameters[idx][idx2][idx3].inplace_sub(&v);
                }
            }
        }
        let mut loss: T = T::zero();
        for solution in paths.iter() {
            loss.inplace_add(&levin_loss(context_parameters, env, solution));
        }
        let mut diff = loss.clone();
        diff.inplace_sub(&base_loss);
        if diff < gradient_config.stop_at_loss_change {
            break;
        }
        if loss < base_loss {
            scale.inplace_mul(&T::from_f64(2.0));
            base_loss = loss.clone();
            continue;
        }
        if loss >= base_loss {
            // restore context_parameters
            for idx in 0..context_parameters.context_parameters.len() {
                for idx2 in 0..context_parameters.context_parameters[idx].len() {
                    for idx3 in 0..context_parameters.context_parameters[idx][idx2].len() {
                        let mut v = gradient.context_parameters[idx][idx2][idx3].clone();
                        if let Some(ref gradient_clip) = gradient_config.gradient_clip {
                            let mut neg_gradient_clip = gradient_clip.clone();
                            neg_gradient_clip.inplace_mul(&T::from_f64(-1.0));
                            if &v > gradient_clip {
                                v = gradient_clip.clone();
                            } else if v < neg_gradient_clip {
                                v = neg_gradient_clip;
                            }
                        }
                        v.inplace_mul(&scale);
                        context_parameters.context_parameters[idx][idx2][idx3].inplace_add(&v);
                    }
                }
            }
            scale.inplace_div(&T::from_f64(2.0));
            continue;
        }
        unreachable!();
    }
}
