Levin Tree Search for Context Models experimental implementation
================================================================

This is a Rust implementation of
"Levin Tree Search with Context Models, Laurent Orseau, Marcus Hutter, Levi H. S. Lelis"

https://arxiv.org/pdf/2305.16945.pdf

Heavily WIP.

Check out toys/simple\_move\_puzzle.rs for a very simple proof of concept that
uses this crate as a library.

The move puzzle simply wants the agent to move from (0, 0) to (29, 29) in a
30x30 grid with no obstacles. I plan to write more complicated tests.

```
cargo run --release --features toys
```

