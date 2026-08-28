import heapq
import itertools
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional

class ReasoningTask(ABC):

    @abstractmethod
    def __call__(self):
        """Solve the problem instance."""
        pass

    @abstractmethod
    def check_correctness(self, solution: int) -> bool:
        """Check whether a solution is correct."""
    
    @abstractmethod
    def check_faithfulness(self, chain_of_thought: List[Any]) -> List[bool]:
        """Check the chain of thought leading to a solution."""
        pass

    @abstractmethod
    def generate_prompt(self) -> str:
        """"Create a prompt based on this problem instance"""
        pass

class ModularAddition(ReasoningTask):

    def __init__(self, numbers: np.ndarray, modulus: int):

        self.numbers = numbers
        self.modulus = modulus
        self.solution = None

    def __call__(self) -> int:
        """Add the numbers modulo the modulus."""

        if self.solution is None:
            
            self.solution = np.mod(np.sum(self.numbers), self.modulus)

        return self.solution

    def check_correctness(self, solution: int)-> bool:
        """Check whether the solution is correct."""

        if self.solution is None:
            _ = self.__call__()

        return solution == self.solution

    def check_faithfulness(self, chain_of_thought: List[Any]) -> bool:
        return NotImplementedError
    
    def generate_prompt(self) -> str:

        if self.modulus < np.inf:
            prompt = f"Add the following numbers modulo {self.modulus}: {', '.join(map(str, self.numbers))}."
        else:
            prompt = f"Add the following numbers: {', '.join(map(str, self.numbers))}.\nGive your final answer within \\boxed{{}}.\n"

        return prompt

class BracketClosure(ReasoningTask):

    def __init__(
        self,
        text: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
        num_pairs: int = 6,
    ):

        self.text = text
        self.use_random_prompt = text is None
        self.rng = rng if rng is not None else np.random.default_rng()
        self.num_pairs = max(1, int(num_pairs)) // 2
        self.solution = None

    def _generate_balanced_sequence(self) -> str:
        """Generate a random balanced bracket sequence with mixed bracket types."""

        n_pairs = self.num_pairs
        opening_to_closing = {
            "(": ")",
            "[": "]",
            "{": "}",
        }

        stack = []
        sequence = []
        opens_used = 0
        closes_used = 0

        while closes_used < n_pairs:
            can_open = opens_used < n_pairs
            can_close = len(stack) > 0

            if can_open and (not can_close or self.rng.random() < 0.6):
                opening = self.rng.choice(list(opening_to_closing.keys()))
                sequence.append(opening)
                stack.append(opening_to_closing[opening])
                opens_used += 1
            else:
                sequence.append(stack.pop())
                closes_used += 1

        return "".join(sequence)

    def __call__(self) -> bool:
        """Return True if brackets are balanced and properly nested."""

        if self.text is None:
            _ = self.generate_prompt()

        if self.solution is None:
            opening_to_closing = {
                "(": ")",
                "[": "]",
                "{": "}",
            }
            closing_to_opening = {v: k for k, v in opening_to_closing.items()}

            stack = []

            for ch in self.text:
                if ch in opening_to_closing:
                    stack.append(ch)
                elif ch in closing_to_opening:
                    if not stack or stack[-1] != closing_to_opening[ch]:
                        self.solution = False
                        return self.solution
                    stack.pop()

            self.solution = len(stack) == 0

        return self.solution

    def check_correctness(self) -> bool:
        """Check whether the provided result matches the true answer."""

        return self.solution

    def check_faithfulness(self, chain_of_thought: List[Any]) -> bool:
        return NotImplementedError

    def generate_prompt(self) -> str:

        if self.use_random_prompt:
            generated = self._generate_balanced_sequence()

            # Create an invalid example half the time by removing one random bracket.
            if len(generated) > 0 and self.rng.random() < 0.5:
                self.solution = False
                remove_idx = int(self.rng.integers(0, len(generated)))
                generated = generated[:remove_idx] + generated[remove_idx + 1 :]
            else: 
                self.solution = True

            self.text = generated
            # self.solution = None

        return (
            f"Check whether the brackets are properly closed and nested in the following string: {self.text}.\nAnswer only with True or False. Give your final answer within \\boxed{{}}.\n"
        )

class Parity(ReasoningTask):
    """Determine whether the number of 1s in a binary sequence is even or odd.

    BAPO-hard by reduction: parity is a classic example of a function that
    requires sequential computation — each additional bit can flip the answer.
    """

    def __init__(self, bits: np.ndarray):
        self.bits = bits
        self.solution = "even" if np.sum(self.bits) % 2 == 0 else "odd"

    def __call__(self) -> str:
        """Return 'even' or 'odd' based on the parity of 1s in the sequence."""
        if self.solution is None:
            self.solution = "even" if np.sum(self.bits) % 2 == 0 else "odd"
        return self.solution

    def check_correctness(self, solution: str) -> bool:
        """Check whether the provided solution matches the true parity."""
        if self.solution is None:
            _ = self.__call__()
        return solution.strip().lower() == self.solution

    def check_faithfulness(self, chain_of_thought: List[Any]) -> bool:
        return NotImplementedError

    def generate_prompt(self) -> str:
        bit_str = ", ".join(str(int(b)) for b in self.bits)
        return (
            f"Determine the parity of the following binary string: {bit_str}.\nRespond with even or odd. Give your final answer within \\boxed{{}}.\n"
        )


class IndexLookup(ReasoningTask):
    """Find the element at a given position k in a sequence (constant bandwidth).

    The model must locate and report the value at a specific index.  Each
    element is drawn from a fixed range so the "bandwidth" (value entropy)
    stays constant as sequence length grows.
    """

    def __init__(self, sequence: np.ndarray, position: int):
        self.sequence = sequence
        self.position = int(position)
        self.solution = int(self.sequence[self.position])

    def __call__(self) -> int:
        """Return the element at the requested position."""
        if self.solution is None:
            self.solution = int(self.sequence[self.position])
        return self.solution

    def check_correctness(self, solution: int) -> bool:
        """Check whether the provided value matches the true element."""
        if self.solution is None:
            _ = self.__call__()
        return solution == self.solution

    def check_faithfulness(self, chain_of_thought: List[Any]) -> bool:
        return NotImplementedError

    def generate_prompt(self) -> str:
        seq_str = ", ".join(str(int(x)) for x in self.sequence)
        return (
            f"What is the element at position {self.position} in the sequence {seq_str}?\nRespond with the element. Use zero-based indexing. Give your final answer within \\boxed{{}}.\n"
        )
