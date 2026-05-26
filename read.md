I studied the simulator stack. Here’s the map.

Big Picture
The project simulates serverless autoscaling with a vector SPSA/NSGD learner. The learned parameter is:

theta = [theta_stock, theta_idle, theta_exp]

theta_stock: how many init-free/warmable instances to maintain or spawn.
theta_idle: idle threshold that triggers preemptive provisioning.
theta_exp: controls expiration rate via expiration_process.rate = theta_exp / K_exp.
Core files:

simulate.py: main runner, single-node simulator, and newer DAG/graph simulator.
ServerlessSimulatorfixarrival.py: older/simpler experiment runner, mostly single-node plus partial DAG arrival-rate aggregation.
Algorithm.py: vectorial SPSA autoscaling algorithm.
FunctionInstance.py: lifecycle/state machine for one function instance.
utils.py: enums for function/system state.
Event Flow
ServerlessSimulator.generate_trace() is the event loop. At each step it picks whichever happens next:

External arrival.
Function instance transition.
Arrivals are handled by pool availability:

idle instance exists: warm_start_arrival()
no idle, init-free exists: init_free_arrival()
neither exists: cold_start_arrival()
After each event, the simulator calls:

self.update_state()
self.autoscaler.simulate_step(self.state, self)
So the learner updates from the event-by-event system state.

Function Lifecycle
A FunctionInstance starts as COLD, then can become:

COLD -> INIT_FREE -> IDLE_ON -> BUSY -> IDLE_ON -> COLD

or for a cold request:

COLD -> INIT_RESERVED -> BUSY -> IDLE_ON -> COLD

For queued-on-init-free requests:

INIT_FREE(reserved) -> BUSY -> IDLE_ON

Algorithm
Algorithm.py implements two SPSA phases per iteration:

Simulate with theta + perturbation
Simulate with theta - perturbation
Estimate gradient from cost difference
Update theta using SGD, Adam, or RMSProp
Cost is:

dot(state, weights) + w_rej * rejected
with default weights roughly:

cold: 0
idle-on: 2
busy: 1
initializing: 5
reserved: 100
rejection penalty: 200
Important Difference Between Files
simulate.py has the newer graph-level simulation: GraphServerlessSimulator. It uses one external arrival stream at the root node, routes completed requests downstream using the DAG transition matrix, and shares max_concurrency across all nodes.

ServerlessSimulatorfixarrival.py appears older: it computes expected arrival rates per DAG node and runs each node separately. That is less realistic for shared-cluster DAG behavior because node executions are not causally linked by actual completions.

So for DAG workflow experiments, simulate.py looks like the file to trust more.

Things I Noticed
A few likely bug/risk areas:

In FunctionInstance.transition_state_to(), the assert only allows INIT_FREE or INIT_RESERVED, but the later code checks state == IDLE_ON; that branch is unreachable.
In get_request_custom_states(), rejected requests are accidentally added to state_req_warm instead of state_req_rejs.
cold_service_process in both simulator constructors is created without the seeded RNG, unlike warm/cold-start/arrival/expiration, so reproducibility may be slightly inconsistent.
ServerlessSimulatorfixarrival.py has a typo print: "Loadin Dag graph from : {dag_path}" is missing the f.
simulate.py and ServerlessSimulatorfixarrival.py duplicate a lot of code; fixes in one may not exist in the other.
Net: the code is an event-driven serverless lifecycle simulator wrapped around vector SPSA learning, with simulate.py now containing the more complete shared-cluster DAG implementation.





4:08 PM

Autoscaling-1 (2).pdf
this is the graph extended of this paper
do you think its ok?
4:16 PM


6 previous messages
Short answer: yes, the graph extension is a reasonable idea, and the newer implementation in simulate.py (line 803) is much closer to what I’d consider “correct” than the older arrival-rate-splitting approach in ServerlessSimulatorfixarrival.py.

But I would not yet claim it is theoretically equivalent to the paper without tightening a few modeling choices.

What Looks Good
The graph simulator does the right high-level thing:

One external arrival process enters the root node.
When a node finishes service, the request is routed downstream using the DAG transition probabilities.
All nodes share one cluster-wide max_concurrency.
Each DAG node has its own function pools, cold/warm/init lifecycle, and autoscaler.
Graph-level state aggregates all nodes through _global_state() in simulate.py (line 911).
That is much better than computing expected arrival rates per node and running independent simulations, because the latter loses causality, shared-capacity contention, and actual workflow timing.

Main Concern
The biggest issue is learning: _advance_all_autoscalers() updates every node autoscaler on every graph event using the same global state and same global rejection signal:

simulate.py (line 926)

That means each node’s SPSA gradient is estimated while other nodes’ parameters are also changing. So the gradient signal for node A is confounded by simultaneous perturbations in nodes B, C, etc.

This can still work experimentally, but it is no longer the clean 3-parameter NSGD from the paper. It becomes a multi-agent/global-objective variant.

Cleaner options:

Centralized graph NSGD: one learner optimizes a big vector
theta = [theta_stock_A, theta_idle_A, theta_exp_A, theta_stock_B, ...]

Shared-policy NSGD: one theta is shared by all nodes, if all functions should follow the same policy.

Local node NSGD: each node learns from local state/local cost, while the graph simulator only handles routing and shared capacity.

For a paper extension, option 1 is probably the strongest and most defensible.

Other Notes
The shared cluster logic itself looks sensible:

current_concurrency() sums active/reserved/booked work across nodes.
current_cold_servers() uses total allocated servers across all nodes.
downstream routing only happens after a busy request completes, which is realistic for sequential workflow stages.
One subtle modeling question: the algorithm’s cost samples are event-based, while the graph simulator also computes a time-average graph cost separately. If your paper extension discusses continuous-time cost, you may want the learner to optimize the same time-average cost you report.

So: conceptually yes, it is ok and promising. I’d just avoid saying “this directly preserves the paper’s convergence guarantee” unless you reformulate the graph system as one joint Markov process with one joint parameter vector.





4:17 PM
also the maximum concurrent should be shared for the whole cluster (all grpah has its maximum concurrent request)
.
4:18 PM

Yes, exactly. For the graph/workflow version, max_concurrency should be cluster-global, not per node.

So if max_concurrency = 50, that means across the whole DAG:

running + init_reserved + booked_init_free <= 50
summed over all graph nodes, not 50 for each node.

In your current newer simulate.py (line 885), this is already modeled correctly:

def current_concurrency(self):
    return sum(
        sim.running_count + sim.init_reserved_count + sim.init_free_booked_count
        for sim in self.node_sims.values()
    )
and available cold capacity is also shared:

used_servers = sum(sim.server_count for sim in self.node_sims.values())
return max(0, self.maximum_concurrency - used_servers)
So the graph-level simulator enforces one shared cluster budget.

But ServerlessSimulatorfixarrival.py is not okay for that requirement if it runs each node independently with computed arrival rates. That effectively gives each node its own concurrency limit, unless manually adjusted, which is not the same as a shared cluster.

So my recommendation is: use simulate.py’s GraphServerlessSimulator as the graph extension baseline. The older ServerlessSimulatorfixarrival.py should either be retired or updated to use the same shared-cluster event-based logic.





4:18 PM
no tell me how a job would be rejected based on the code
only on the entry node?
4:20 PM

No. In the newer graph simulator in simulate.py, a job can be rejected at any node, not only the entry/root node.

The rejection happens inside each node simulator’s arrival handlers:

cold_start_arrival()
warm_start_arrival()
init_free_arrival()
Each one checks:

if self.has_reached_max_concurrency():
    self.total_reject_count += 1
    self.job_rejected = True
    ...
    return
For graph mode, self.has_reached_max_concurrency() delegates to the shared cluster:

def has_reached_max_concurrency(self):
    if self.cluster is not None:
        return self.cluster.has_reached_max_concurrency()
and the cluster computes:

def has_reached_max_concurrency(self):
    return self.current_concurrency() >= self.maximum_concurrency
where current_concurrency() sums all nodes:

sum(
    sim.running_count + sim.init_reserved_count + sim.init_free_booked_count
    for sim in self.node_sims.values()
)
So a downstream job can be rejected too. Example:

External request enters root node A.
It finishes at A.
_route_downstream() sends it to node B.
_dispatch_arrival("B", t) tries to admit it.
If the whole cluster is already at max_concurrency, node B rejects it.
So rejection is cluster-wide but triggered at whichever node receives the arrival. External arrivals can be rejected at the entry node; internal routed arrivals can be rejected at downstream nodes.





4:20 PM
ok good
also each job is being processed and the other jobs in the queue are going together yes?
4:20 PM

Mostly yes, but with an important detail: the code is event-driven, so jobs are not processed in batches. They progress independently, and the simulator always jumps to the next event in time.

In graph mode:

A job enters the root node.
When it starts running on an instance, it has its own completion time.
Other jobs can arrive and run on other instances while this one is still running.
When a job finishes at one node, _route_downstream() immediately creates an internal arrival for the next node at the same timestamp.
The simulator then continues processing the earliest next event among all nodes.
So jobs “go together” in the sense that many jobs can be active across the graph at the same simulated time, sharing the same cluster concurrency limit.

But there is not a global FIFO queue where many waiting jobs are advanced together. The only “queued” case in this code is when a request is assigned to an INIT_FREE instance that is still initializing:

self.init_free_booked_count += 1
self.queued_jobs_count += 1
That queued job waits for that specific instance to finish initialization, then becomes BUSY.

So:

Concurrent processing? yes.
Shared cluster limit? yes.
Workflow routing after completion? yes.
Batch movement of all queued jobs together? no.
Central queue across the whole graph? no.
Each job follows its own path through the DAG, and the simulator interleaves all jobs by event time.





4:21 PM
yes yes I know but a node can process 4 jobs for example
I mean lets say job a comes with servie time 3
after 2 second job b comes with 5 service time
the node should process them both after two second because both has arrived now
correct?
is the code doing thaT?
4:22 PM

Yes, if the node has capacity/instances, the code does that.

Example:

t = 0: job A arrives, service time = 3
t = 2: job B arrives, service time = 5
If job A is already running and job B arrives at t = 2, the simulator does not wait for A to finish. It handles B as a new arrival. If the node has an idle/warm instance, B starts immediately at t = 2.

So from:

t = 2 to t = 3
both A and B are running concurrently.

Then:

A finishes at t = 3
B finishes at t = 7
The code supports this because each FunctionInstance has its own next_departure, and the simulator always finds the next event among all instances:

server_next_transitions = np.array([s.get_next_transition_time(t) for s in self.servers])
So yes: a node can process multiple jobs at the same time, as long as there are enough instances and the shared cluster max_concurrency is not reached.

Small nuance: one function instance processes one job at a time. So “node processes 4 jobs” means the node has 4 busy instances, not one instance handling 4 jobs.





4:22 PM
ok perfect 
now imagine that we are new to the code and wants to read it
explain the code to us with detail
4:24 PM


Absolutely. Let’s read it like we just walked into the room and found a simulator with a lot of moving parts on the table.

The code is trying to simulate a serverless/FaaS platform where requests arrive, function instances are created, warmed, used, idled, expired, and sometimes rejected. On top of that, an online learning algorithm adjusts autoscaling parameters while the simulation is running.

The newer important file is probably simulate.py, while ServerlessSimulatorfixarrival.py is an older/partial graph version.

Core Idea
The simulator has two layers:

Serverless system layer
This models arrivals, function instances, cold starts, warm starts, queues, running jobs, idle instances, and rejections.

Learning/autoscaling layer
This learns the vector:

theta = [theta_stock, theta_idle, theta_exp]
Meaning:

theta_stock: how many pre-warmed/init-free instances to keep or create.
theta_idle: idle threshold that triggers more provisioning.
theta_exp: expiration rate for idle instances.
The learning algorithm lives in Algorithm.py.

Function States
The states are defined in utils.py:

class FunctionState(Enum):
    COLD = 0
    INIT_FREE = 1
    INIT_RESERVED = 2
    IDLE_ON = 3
    BUSY = 4
A function instance can be:

COLD: inactive, not initialized.
INIT_FREE: being initialized, but not assigned to a request.
INIT_RESERVED: being initialized for a specific request.
IDLE_ON: warm and ready.
BUSY: processing a request.
The system-level state is:

class SystemState(Enum):
    COLD = 0
    IDLE_ON = 1
    BUSY = 2
    INITIALIZING = 3
    INIT_RESERVED = 4
This is the aggregated state passed to the cost function.

FunctionInstance
FunctionInstance.py represents one function container/instance.

When a FunctionInstance is created, it gets:

cold start process
warm service process
cold service process
expiration process
It also immediately generates its first initialization completion time:

self.generate_init_departure(t)
self.update_next_termination()
Important transitions:

COLD -> INIT_FREE
COLD -> INIT_RESERVED
INIT_FREE -> IDLE_ON
INIT_RESERVED -> BUSY
IDLE_ON -> BUSY
BUSY -> IDLE_ON
IDLE_ON -> COLD
A warm arrival calls:

arrival_transition(t)
If the instance is IDLE_ON, it becomes BUSY and gets a warm-service departure time.

If the instance is INIT_FREE, it gets reserved for a queued job.

Single-Node Simulator
The main single-node simulator class is ServerlessSimulator.

It tracks counters like:

self.running_count
self.idle_count
self.init_free_count
self.init_reserved_count
self.init_free_booked_count
self.server_count
These counters are the simulator’s live view of the node.

A request arrival is handled by one of three methods.

If warm instance exists:

warm_start_arrival()
The request starts immediately on an idle instance.

If no warm instance exists but init-free exists:

init_free_arrival()
The request reserves an initializing instance and waits until initialization finishes.

If neither exists:

cold_start_arrival()
The simulator creates an INIT_RESERVED instance for the request, and may also create extra INIT_FREE instances according to theta_stock.

The Event Loop
The most important method is:

generate_trace()
This is a discrete-event simulation loop. It does not advance time second by second. Instead, it jumps to the next interesting event.

At each loop, it checks:

When is the next request arrival?
When is the next function transition?
Then it processes whichever happens first.

So if:

Job A arrives at t=0 and runs until t=3
Job B arrives at t=2
the simulator jumps:

t=0 -> t=2 -> t=3
At t=2, B can start even while A is still running, assuming capacity exists.

That is why concurrency works.

Rejection Logic
A request is rejected when the shared concurrency limit is reached.

In single-node mode:

return self.current_concurrency() >= self.maximum_concurrency
where concurrency is:

running_count + init_reserved_count + init_free_booked_count
In graph mode, the node delegates that check to the cluster:

if self.cluster is not None:
    return self.cluster.has_reached_max_concurrency()
So in graph mode, rejection is based on the whole DAG/cluster, not just one node.

A request can be rejected at the root node or at a downstream node.

Autoscaling Algorithm
Algorithm.py contains VectorialAutoScalingAlgorithm.

It implements SPSA-style learning.

Each iteration does:

simulate with theta + perturbation
simulate with theta - perturbation
compare costs
estimate gradient
update theta
The algorithm produces:

theta_step = [theta_stock_step, theta_idle_step, theta_exp_step]
The first two are stochastically rounded because they represent integer counts/thresholds.

The expiration parameter is continuous:

simulator.expiration_process.rate = theta_exp_step / K_exp
Cost is computed from the system state:

np.dot(self.state, self.weights) + self.w_rej * self.has_rejected_job
So the learner tries to minimize resource cost plus rejection penalty.

Graph Extension
The graph extension is in:

class GraphServerlessSimulator
in simulate.py (line 803).

This models a workflow/DAG.

Each DAG node gets its own ServerlessSimulator, meaning each node has its own function pools and autoscaler.

But the graph simulator owns:

the external arrival stream
the DAG transition matrix
shared cluster concurrency
downstream routing
global event loop
When a job finishes at a node, this happens:

self._route_downstream(node_name, t)
It samples the transition matrix and sends the job to the next node at the same timestamp.

So a job path looks like:

external arrival -> root node -> maybe node B -> maybe node C -> exit
Shared Cluster Capacity
This is important.

Graph mode correctly shares capacity across all nodes:

def current_concurrency(self):
    return sum(
        sim.running_count + sim.init_reserved_count + sim.init_free_booked_count
        for sim in self.node_sims.values()
    )
And total allocated servers are also shared:

used_servers = sum(sim.server_count for sim in self.node_sims.values())
return max(0, self.maximum_concurrency - used_servers)
So if max_concurrency = 50, that is 50 total across the whole graph, not 50 per node.

Important Design Point
The graph code currently updates all node autoscalers at every graph event using the same global state:

sim.autoscaler.simulate_step(global_state, sim)
That means each node learns from the whole cluster state, not only its own local state.

This is a valid experimental design if you want global-cost learning, but it is not exactly the original single-function paper model anymore. It is a graph/workflow extension.

Old vs New Graph Approach
The old approach in ServerlessSimulatorfixarrival.py computes expected arrival rates for each node and runs each node separately.

That is weaker because it loses:

real routing timing
shared concurrency contention
downstream rejection behavior
causal job completion before next-node arrival
The newer GraphServerlessSimulator in simulate.py is the better model.

Mental Model
If we simplify the whole system:

Request arrives
    |
    v
Is warm instance available?
    yes -> run immediately
    no
      |
      v
Is init-free available?
    yes -> reserve it, wait until initialized, then run
    no
      |
      v
Can cluster allocate more?
    yes -> cold start / init-reserved
    no -> reject
After every event:

update counters
compute system state
charge cost
maybe update theta
In graph mode, after a request finishes at one node:

sample DAG transition
send request to next node
repeat same arrival logic there
So the simulator is basically a living queueing system plus an online learner watching and tuning it.