'''Basic McCulloch-Pitts (MC) neuron.

The neuron computes a weighted sum of inputs. It holds its output at 0
while the sum is below the threshold, and fires (outputs 1) as soon as the
sum reaches or exceeds the threshold.
'''

from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass
class MCNeuron:
    '''A simple threshold neuron.'''

    weights: Sequence[float]
    threshold: float = 1.0
    reset_after_fire: bool = True
    potential: float = field(default=0.0, init=False)
    fired: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.weights = list(self.weights)

    def reset(self) -> None:
        '''Clear the neuron state.'''
        self.potential = 0.0
        self.fired = False

    def step(self, inputs: Iterable[float]) -> int:
        '''Process one input pattern and return 1 if the neuron fires.'''
        inputs = list(inputs)
        if len(inputs) != len(self.weights):
            raise ValueError(
                f'expected {len(self.weights)} inputs, got {len(inputs)}'
            )

        self.potential = sum(w * x for w, x in zip(self.weights, inputs))
        if self.potential >= self.threshold:
            self.fired = True
            if self.reset_after_fire:
                self.potential = 0.0
            return 1

        self.fired = False
        return 0


def main() -> None:
    neuron = MCNeuron(weights=[1, 1], threshold=2)

    patterns = [
        [1, 0],
        [0, 1],
        [1, 1],
    ]

    for pattern in patterns:
        output = neuron.step(pattern)
        state = 'FIRES' if output else 'holds'
        print(f'input={pattern} -> output={output} ({state})')


if __name__ == '__main__':
    main()
