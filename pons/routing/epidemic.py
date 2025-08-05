from .router import Router
import random

class EpidemicRouter(Router):
    """
    Epidemic routing with probabilistic forwarding:
      p1: probability to forward directly to destination
      p2: probability to forward to a peer in store
      p3: probability to broadcast as fallback (if needed)
    """

    def __init__(
        self,
        scan_interval: float = 2.0,
        capacity: int = 0,
        apps: list = None,
        p1: float = 0.1,
        p2: float = 0.2,
        p3: float = 0.5, 
        energy_thresh:float = 0.0,
        pop_thresh:float = 0.0
    ):
        super(EpidemicRouter, self).__init__(scan_interval, capacity, apps)
        # forwarding probabilities
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.energy_thresh = energy_thresh
        self.pop_thresh   = pop_thresh    

    def __str__(self):
        return f"EpidemicRouter(p1={self.p1}, p2={self.p2}, p3={self.p3})"

    def add(self, msg):
        """Add new message to store and attempt forwarding."""
        if self.store_add(msg):
            self.forward(msg)

    def forward(self, msg):
        node = self.netsim.nodes[self.my_id]
        curr_energy = node.energy
        curr_pop    = node.popularity
        """Probabilistic forwarding logic."""
        # Direct-to-destination with probability p1
        if (curr_energy >= self.energy_thresh and curr_pop >= self.pop_thresh and msg.dst in self.peers and not self.msg_already_spread(msg, msg.dst) and random.random() <= self.p1):
                self.netsim.routing_stats["started"] += 1
                self.send(msg.dst, msg)
                self.remember(msg.dst, msg.unique_id())
                self.store_del(msg)
                return

        # Peer-to-peer forwarding with probability p2
        for peer in self.peers:
            if not self.msg_already_spread(msg, peer):
                if (curr_energy >= self.energy_thresh and curr_pop >= self.pop_thresh and random.random() <= self.p2):
                    self.netsim.routing_stats["started"] += 1
                    self.send(peer, msg)
                    self.remember(peer, msg.unique_id())
        # Optional broadcast fallback with p3
        # Uncomment below to enable broadcast fallback
        if (curr_energy >= self.energy_thresh  and curr_pop >= self.pop_thresh and random.random() <= self.p3):
            for peer in self.peers:
                if not self.msg_already_spread(msg, peer):
                    self.netsim.routing_stats["started"] += 1
                    self.send(peer, msg)
                    self.remember(peer, msg.unique_id())

    def on_peer_discovered(self, peer_id):
        """When a new peer appears, attempt to forward all stored messages."""
        for msg in list(self.store):
            # remove expired
            if msg.is_expired(self.netsim.env.now):
                self.store_del(msg)
            else:
                if not self.msg_already_spread(msg, peer_id):
                    self.forward(msg)

    def on_msg_received(self, msg, remote_id, was_known):
        """Handle reception of a message: store and forward if new and not destined for this node."""
        if not was_known and msg.dst != self.my_id:
            if self.store_add(msg):
                self.forward(msg)