#Here the DTN protocol is implemented

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Set, Optional
import time

@dataclass 
class DTNMessage:
    id: str
    source_id: str
    destination_id: str
    data: str
    generation_time: float
    hop_count: int = 0
    ttl: float = 3600  # Time-to-live in seconds (this is 1 hour)
    priority: int = 1  # Default priority
    size: int = 100  # Size in bytes

    def is_expired(self, current_time: float) -> bool:
        return (current_time - self.generation_time) > self.ttl
    
    def get_AoI(self, current_time: float) -> float:
        return current_time - self.generation_time
    
class DTNProtocol(ABC):  # Abstract class for DTN Protocols

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.message_history: Set[str] = set()  # Track seen messages

    @abstractmethod
    def should_forward(self, message: DTNMessage, to_node_id: str, current_time: float) -> bool:
        pass

    @abstractmethod
    def select_messages_to_forward(self, messages: List[DTNMessage], to_node_id: str, current_time: float) -> List[DTNMessage]:
        pass

    @abstractmethod
    def get_protocol_name(self) -> str:
        pass

    def has_seen_message(self, message_id: str) -> bool:
        # Check if the message has been seen before
        return message_id in self.message_history
    
    def mark_seen_message(self, message_id: str):
        # Mark the message as seen
        self.message_history.add(message_id)

class EpidemicProtocol(DTNProtocol):
    # Implements Epidemic Routing Protocol, this protocol forwards all messages that the other node has not seen yet.

    def should_forward(self, message: DTNMessage, to_node_id: str, current_time: float) -> bool:
        # In Epidemic routing, forward if message is not expired
        return not message.is_expired(current_time)
    
    def select_messages_to_forward(self, messages: List[DTNMessage], to_node_id: str, current_time: float) -> List[DTNMessage]:
        # Forward all messages that are not expired
        return [msg for msg in messages if self.should_forward(msg, to_node_id, current_time)]
    
    def get_protocol_name(self) -> str:
        return "Epidemic"
    
class SprayAndWaitProtocol(DTNProtocol):
    # Implements Spray and Wait Protocol, this protocol forwards a limited number of copies of each message.

    def __init__(self, node_id: str, max_copies: int = 5):
        super().__init__(node_id)
        self.max_copies = max_copies
        self.message_copies: Dict[str, int] = {}  # Track number of copies for each message

    def should_forward(self, message: DTNMessage, to_node_id: str, current_time: float) -> bool:
        # Forward if we have remaining copies and the message is not expired
        if message.is_expired(current_time):
            return False
        return self.message_copies.get(message.id, self.max_copies) > 0
    
    def select_messages_to_forward(self, messages: List[DTNMessage], to_node_id: str, current_time: float) -> List[DTNMessage]:
        selected_messages = []
        for msg in messages:
            if self.should_forward(msg, to_node_id, current_time):
                selected_messages.append(msg)
                # Decrement the number of copies left
                self.message_copies[msg.id] = self.message_copies.get(msg.id, self.max_copies) - 1
        return selected_messages
    
    def get_protocol_name(self) -> str:
        return "Spray and Wait"

class DirectDeliveryRouting(DTNProtocol):
    """Direct delivery - only forward to destination"""
    
    def should_forward(self, message: DTNMessage, to_node_id: str, current_time: float) -> bool:
        return (not message.is_expired(current_time) and 
                (message.destination_id == to_node_id or message.destination_id == "broadcast"))
    
    def select_messages_to_forward(self, messages: List[DTNMessage], to_node_id: str, 
                                 current_time: float) -> List[DTNMessage]:
        return [msg for msg in messages if self.should_forward(msg, to_node_id, current_time)]
    
    def get_protocol_name(self) -> str:
        return "DirectDelivery"