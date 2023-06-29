use trlevin::trlevin::*;

#[derive(Clone, Ord, Debug, Eq, PartialEq, PartialOrd)]
struct Environment {
    width: i64,
    height: i64,
}

#[derive(Clone, Ord, Debug, Eq, PartialEq, PartialOrd)]
struct State {
    last_move: Move,
    x: i64,
    y: i64,
}

#[derive(Clone, Ord, Debug, Eq, PartialEq, PartialOrd)]
enum Move {
    Up,
    Right,
    Left,
    Down,
}

impl Enumerable for Move {
    fn to_number(&self) -> usize {
        match self {
            Move::Up => 0,
            Move::Right => 1,
            Move::Left => 2,
            Move::Down => 3,
        }
    }
}

impl ContextModelable for Environment {
    type Action = Move;
    type State = State;

    fn is_solution(&self, state: &Self::State) -> bool {
        state.x == 0 && state.y == self.height / 2
    }

    fn possible_actions(&self, state: &Self::State, actions: &mut Vec<Self::Action>) {
        actions.truncate(0);
        if state.x > 0 {
            actions.push(Move::Left);
        }
        if state.y > 0 {
            actions.push(Move::Up);
        }
        if state.x < self.width - 1 {
            actions.push(Move::Right);
        }
        if state.y < self.height - 1 {
            actions.push(Move::Down);
        }
    }

    fn perform_action(
        &self,
        _state: &Self::State,
        action: &Self::Action,
        new_state: &mut Self::State,
    ) {
        new_state.last_move = action.clone();
        match action {
            Move::Up => new_state.y -= 1,
            Move::Right => new_state.x += 1,
            Move::Left => new_state.x -= 1,
            Move::Down => new_state.y += 1,
        }
    }

    fn active_contexts(&self, state: &Self::State, active_contexts: &mut Vec<usize>) {
        active_contexts.truncate(0);

        // last move
        active_contexts.push(state.last_move.to_number());
        // wall on left?
        if state.x > 0 {
            active_contexts.push(1);
        } else {
            active_contexts.push(0);
        }
        // wall on right?
        if state.x < self.width - 1 {
            active_contexts.push(1);
        } else {
            active_contexts.push(0);
        }
        // wall on up?
        if state.y > 0 {
            active_contexts.push(1);
        } else {
            active_contexts.push(0);
        }
        // wall on down?
        if state.y < self.height - 1 {
            active_contexts.push(1);
        } else {
            active_contexts.push(0);
        }
    }
}

fn main() {
    let env = Environment {
        width: 30,
        height: 30,
    };

    let mut params: ContextModelParameters<OrderedFloat<f64>> =
        ContextModelParameters::new(&[4, 2, 2, 2, 2], 4);
    let initial_state: State = State {
        last_move: Move::Up,
        x: 0,
        y: 0,
    };

    let mut past_solutions = vec![];
    let mut gradient = params.clone();

    loop {
        let mut gstate = gradient.new_gradient_state();
        println!("{:?}", params);
        match levin_search(&env, initial_state.clone(), &params, Some(1000000)) {
            Some(LevinSearchSolution::Path(path, _final_state)) => {
                println!("Path len={:?}", path.len());
                let loss = levin_loss(&params, &env, &path);
                past_solutions.push(path);
                println!("Loss={}", loss);

                gradient.set_zero();
                for s in past_solutions.iter() {
                    loss_gradient(&params, &mut gradient, &mut gstate, &env, s);
                }
                let conf =
                    GradientConfig::new().do_line_search(OrderedFloat(1.0), &env, &past_solutions);

                apply_gradient(&mut params, &gradient, &conf);
            }
            None => {
                println!("No solution found");
                break;
            }
        }
    }
}
