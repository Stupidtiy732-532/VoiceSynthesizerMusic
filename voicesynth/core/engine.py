
from abc import ABC, abstractmethod


class SynthEngine(ABC):

    @abstractmethod
    def synthesize(self, score, voicebank=None):

        pass