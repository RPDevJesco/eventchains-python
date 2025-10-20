"""
EventChains - A Universal Design Pattern for Sequential Workflows

EventChains is a design pattern for building sequential workflows with composable middleware.
It provides a structured approach to building workflows where:
- Events represent individual steps in a process
- Context carries shared state between events
- Middleware adds reusable behaviors around event execution
- Chain orchestrates the sequential flow

Example:
    from eventchains import EventChain, ChainableEvent, EventContext, Result
    
    class MyEvent(ChainableEvent):
        def execute(self, context):
            value = context.get('input')
            context.set('output', value * 2)
            return Result.ok()
    
    chain = EventChain()
    chain.add_event(MyEvent())
    
    context = EventContext({'input': 5})
    result = chain.execute(context)
    print(context.get('output'))  # 10
"""

__version__ = "1.0.0"
__author__ = "EventChains Contributors"

from .chain import EventChain, FaultTolerance
from .context import EventContext
from .event import ChainableEvent
from .middleware import Middleware
from .result import Result

__all__ = [
    'EventChain',
    'EventContext',
    'ChainableEvent',
    'Middleware',
    'Result',
    'FaultTolerance',
]
