/**
 * GraphALP Live Interactive Simulator
 * Pure vanilla JavaScript implementations of graph algorithms, samplers, and canvas visualizers.
 */

// Global State
let simulatorState = {
    graphType: 'grid_cols',          // 'grid_cols', 'grid_rows', 'grid_circ1', 'grid_circ2', 'clustered'
    samplerType: 'harmonic_greedy',   // 'random', 's2', 'harmonic_greedy'
    propagatorType: 'harmonic',      // 'harmonic', 'mincut' (spectral removed)
    
    nodes: [],                       // List of node objects: { id, x, y, label, isQueried, predProb }
    edges: [],                       // Adjacency list: array of { u, v }
    adjList: {},                     // Node ID -> list of neighbor IDs
    labeledSet: {},                  // Node ID -> label (0 or 1)
    featPropFeatures: [],            // Precomputed feature representations for FeatProp
    
    currentStep: 0,
    history: [],                     // Array of risk values for chart plotting
    isPlaying: false,
    playInterval: null,
    
    hoveredNode: null,
    logFeed: []
};

// Canvas Setup
let canvas, ctx;
const CANVAS_WIDTH = 600;
const CANVAS_HEIGHT = 500;

// Initialize Simulator on Load
document.addEventListener('DOMContentLoaded', () => {
    canvas = document.getElementById('simulator-canvas');
    if (canvas) {
        ctx = canvas.getContext('2d');
        // Handle Canvas Clicks (Manual Querying)
        canvas.addEventListener('click', handleCanvasClick);
        canvas.addEventListener('mousemove', handleCanvasMouseMove);
        
        // Initialize Default Graph
        initializeGraph();
    }
    
    // Initialize Interactive Card Demos
    initializeCardDemos();
});

// ---------------------------------------------------------
// 1. Graph Generators
// ---------------------------------------------------------
function initializeGraph() {
    stopAutoPlay();
    simulatorState.nodes = [];
    simulatorState.edges = [];
    simulatorState.adjList = {};
    simulatorState.labeledSet = {};
    simulatorState.currentStep = 0;
    simulatorState.history = [];
    simulatorState.logFeed = [];
    
    const type = simulatorState.graphType;
    const graphNames = {
        'grid_cols': 'Graph 1',
        'grid_rows': 'Graph 2',
        'grid_circ1': 'Graph 3',
        'grid_circ2': 'Graph 4',
        'clustered': 'Graph 5'
    };
    addLog(`Initializing graph: ${graphNames[type] || type.toUpperCase()}`);
    
    if (type.startsWith('grid_')) {
        generateGridGraph(type);
    } else if (type === 'clustered') {
        generateClusteredGraph();
    }
    
    // Build Adjacency List
    for (let i = 0; i < simulatorState.nodes.length; i++) {
        simulatorState.adjList[i] = [];
    }
    simulatorState.edges.forEach(e => {
        simulatorState.adjList[e.u].push(e.v);
        simulatorState.adjList[e.v].push(e.u);
    });
    
    // Precompute FeatProp features
    precomputeFeatPropFeatures();
    
    // Seed 2 initial random nodes (one from class 0, one from class 1)
    seedInitialLabels();
    
    // Run initial propagation
    runPropagation();
    
    // Record starting risk
    recordCurrentRisk();
    
    // Render
    drawGraph();
    updateUIControls();
}

function generateGridGraph(type) {
    const size = 20; // 20x20 Grid (400 nodes)
    const padding = 35;
    const spacingX = (CANVAS_WIDTH - padding * 2) / (size - 1);
    const spacingY = (CANVAS_HEIGHT - padding * 2) / (size - 1);
    
    // Generate Nodes
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const id = y * size + x;
            
            // Layout Coordinates
            const canvasX = padding + x * spacingX;
            const canvasY = padding + y * spacingY;
            
            // Compute Ground Truth Label
            let label = 0;
            if (type === 'grid_cols') {
                label = (x < 10) ? 0 : 1;
            } else if (type === 'grid_rows') {
                label = (y < 10) ? 0 : 1;
            } else if (type === 'grid_circ1') {
                // Diagonal Circles of Radius 7 (Centered at Top-Left and Bottom-Right)
                // Center 1: (0, 0), Center 2: (19, 19)
                const dist1 = Math.sqrt(x*x + y*y);
                const dist2 = Math.sqrt((x-19)*(x-19) + (y-19)*(y-19));
                label = (dist1 <= 7.2 || dist2 <= 7.2) ? 1 : 0;
            } else if (type === 'grid_circ2') {
                // Off-Diagonal Circles of Radius 7 (Centered at Bottom-Left and Top-Right)
                // Center 1: (0, 19), Center 2: (19, 0)
                const dist1 = Math.sqrt(x*x + (y-19)*(y-19));
                const dist2 = Math.sqrt((x-19)*(x-19) + y*y);
                label = (dist1 <= 7.2 || dist2 <= 7.2) ? 1 : 0;
            }
            
            simulatorState.nodes.push({
                id: id,
                x: canvasX,
                y: canvasY,
                gridX: x,
                gridY: y,
                label: label,
                isQueried: false,
                predProb: 0.5
            });
        }
    }
    
    // Generate Grid Edges
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const id = y * size + x;
            // Horizontal Edge
            if (x < size - 1) {
                simulatorState.edges.push({ u: id, v: id + 1 });
            }
            // Vertical Edge
            if (y < size - 1) {
                simulatorState.edges.push({ u: id, v: id + size });
            }
        }
    }
}

function generateClusteredGraph() {
    // 40 nodes: Cluster 0 (0..19) and Cluster 1 (20..39)
    const N = 40;
    
    // Position Cluster 0 on the left, Cluster 1 on the right
    // Cluster 0 center: (180, 250), Cluster 1 center: (420, 250)
    for (let i = 0; i < N; i++) {
        const isCluster1 = i >= 20;
        const centerX = isCluster1 ? 420 : 180;
        const centerY = 250;
        
        // Random position within a circle of radius 85
        const angle = Math.random() * Math.PI * 2;
        const dist = Math.sqrt(Math.random()) * 80;
        const x = centerX + Math.cos(angle) * dist;
        const y = centerY + Math.sin(angle) * dist;
        
        const label = isCluster1 ? 1 : 0;
        
        simulatorState.nodes.push({
            id: i,
            x: x,
            y: y,
            label: label,
            isQueried: false,
            predProb: 0.5
        });
    }
    
    // Generate Stochastic Block Model (SBM) Edges
    // Intra-cluster probability: 0.28
    // Inter-cluster probability: 0.02
    let connected = false;
    while (!connected) {
        simulatorState.edges = [];
        for (let i = 0; i < N; i++) {
            for (let j = i + 1; j < N; j++) {
                const cI = i >= 20;
                const cJ = j >= 20;
                const p = (cI === cJ) ? 0.28 : 0.02;
                if (Math.random() < p) {
                    simulatorState.edges.push({ u: i, v: j });
                }
            }
        }
        
        // Connectivity check via Breadth-First Search
        const visited = new Set();
        const adj = {};
        for (let i = 0; i < N; i++) adj[i] = [];
        simulatorState.edges.forEach(e => {
            adj[e.u].push(e.v);
            adj[e.v].push(e.u);
        });
        
        const queue = [0];
        visited.add(0);
        while (queue.length > 0) {
            const u = queue.shift();
            adj[u].forEach(v => {
                if (!visited.has(v)) {
                    visited.add(v);
                    queue.push(v);
                }
            });
        }
        
        if (visited.size === N) {
            connected = true;
        } else {
            // Secure connection by linking unvisited components
            for (let i = 0; i < N; i++) {
                if (!visited.has(i)) {
                    const visitedArr = Array.from(visited);
                    const target = visitedArr[Math.floor(Math.random() * visitedArr.length)];
                    simulatorState.edges.push({ u: i, v: target });
                    visited.add(i);
                }
            }
            connected = true;
        }
    }
}

function getClumpedSeeds(candidates, count) {
    const candidateIds = new Set(candidates.map(n => n.id));
    const adj = simulatorState.adjList;
    
    // Pick a starting seed node at random
    const startNode = candidates[Math.floor(Math.random() * candidates.length)];
    const clump = [startNode];
    const queue = [startNode.id];
    const visited = new Set([startNode.id]);
    
    while (queue.length > 0 && clump.length < count) {
        const u = queue.shift();
        const neighbors = adj[u] || [];
        
        // Shuffle neighbors to get a natural clump shape
        const shuffledNeighbors = [...neighbors].sort(() => 0.5 - Math.random());
        for (let v of shuffledNeighbors) {
            if (candidateIds.has(v) && !visited.has(v)) {
                visited.add(v);
                clump.push(simulatorState.nodes[v]);
                queue.push(v);
                if (clump.length >= count) break;
            }
        }
    }
    
    // Fallback in case of isolated nodes
    while (clump.length < count) {
        const unvisitedCandidates = candidates.filter(n => !visited.has(n.id));
        if (unvisitedCandidates.length === 0) break;
        const extra = unvisitedCandidates[Math.floor(Math.random() * unvisitedCandidates.length)];
        visited.add(extra.id);
        clump.push(extra);
    }
    
    return clump;
}

function seedInitialLabels() {
    if (simulatorState.propagatorType === 'mincut') {
        // Seed 20 clumped nodes total for Min-Cut
        const class0Nodes = simulatorState.nodes.filter(n => n.label === 0);
        const class1Nodes = simulatorState.nodes.filter(n => n.label === 1);
        
        const seeds0 = getClumpedSeeds(class0Nodes, 10);
        const seeds1 = getClumpedSeeds(class1Nodes, 10);
        
        seeds0.forEach(n => {
            n.isQueried = true;
            simulatorState.labeledSet[n.id] = 0;
        });
        
        seeds1.forEach(n => {
            n.isQueried = true;
            simulatorState.labeledSet[n.id] = 1;
        });
        
        addLog("Seeded min-cut with 20 clumped seed nodes (10 Class 0, 10 Class 1)");
    } else {
        // Find nodes from class 0 and class 1
        const class0Nodes = simulatorState.nodes.filter(n => n.label === 0);
        const class1Nodes = simulatorState.nodes.filter(n => n.label === 1);
        
        // Pick one at random from each
        const seed0 = class0Nodes[Math.floor(Math.random() * class0Nodes.length)];
        const seed1 = class1Nodes[Math.floor(Math.random() * class1Nodes.length)];
        
        seed0.isQueried = true;
        seed1.isQueried = true;
        
        simulatorState.labeledSet[seed0.id] = 0;
        simulatorState.labeledSet[seed1.id] = 1;
        
        addLog(`Seeded initial queries: Node ${seed0.id} (Class 0), Node ${seed1.id} (Class 1)`);
    }
}

// ---------------------------------------------------------
// 2. Label Propagation Solvers
// ---------------------------------------------------------
function runPropagation() {
    const propType = simulatorState.propagatorType;
    
    if (propType === 'harmonic') {
        solveHarmonicLP(simulatorState.labeledSet);
    } else if (propType === 'mincut') {
        solveMinCutLP(simulatorState.labeledSet);
    } else if (propType === 'gcn') {
        solveGCNLP(simulatorState.labeledSet);
    } else {
        // Fallback in case of spectral
        solveHarmonicLP(simulatorState.labeledSet);
    }
}

// A. Harmonic LP (Jacobi Relaxation)
function solveHarmonicLP(labels) {
    const nodes = simulatorState.nodes;
    const adj = simulatorState.adjList;
    
    // Initialize prediction probabilities: Labeled are fixed, unlabeled start at 0.5
    let probs = {};
    nodes.forEach(n => {
        if (n.id in labels) {
            probs[n.id] = labels[n.id];
        } else {
            probs[n.id] = 0.5;
        }
    });
    
    // Jacobi updates (80 steps guarantees tight convergence on a 400-node grid)
    for (let step = 0; step < 80; step++) {
        let nextProbs = { ...probs };
        nodes.forEach(n => {
            if (!(n.id in labels)) {
                const neighbors = adj[n.id];
                if (neighbors.length > 0) {
                    let sum = 0;
                    neighbors.forEach(v => { sum += probs[v]; });
                    nextProbs[n.id] = sum / neighbors.length;
                }
            }
        });
        probs = nextProbs;
    }
    
    // Commit predictions
    nodes.forEach(n => {
        n.predProb = probs[n.id];
    });
}

// B. Min-Cut LP (Breadth-First Ford-Fulkerson Max-Flow Solver)
function solveMinCutLP(labels) {
    const nodes = simulatorState.nodes;
    const N = nodes.length;
    
    // Build s-t flow graph.
    // Source index: N, Sink index: N+1
    const s = N;
    const t = N + 1;
    
    // Identify label classes
    const class0 = Object.keys(labels).filter(id => labels[id] === 0).map(Number);
    const class1 = Object.keys(labels).filter(id => labels[id] === 1).map(Number);
    
    // Build Adjacency representation with capacities
    const capacity = {};
    const flow = {};
    const residualAdj = {};
    
    function addEdge(u, v, cap) {
        if (!residualAdj[u]) residualAdj[u] = [];
        if (!residualAdj[v]) residualAdj[v] = [];
        residualAdj[u].push(v);
        residualAdj[v].push(u);
        capacity[`${u}->${v}`] = cap;
        capacity[`${v}->${u}`] = 0; // Reverse residual edge
        flow[`${u}->${v}`] = 0;
        flow[`${v}->${u}`] = 0;
    }
    
    // Add existing graph edges with capacity 1
    simulatorState.edges.forEach(e => {
        addEdge(e.u, e.v, 1.0);
        addEdge(e.v, e.u, 1.0);
    });
    
    // Connect Source to Class 0 seeds (capacity Infinity)
    class0.forEach(nodeId => {
        addEdge(s, nodeId, 999999);
    });
    
    // Connect Class 1 seeds to Sink (capacity Infinity)
    class1.forEach(nodeId => {
        addEdge(nodeId, t, 999999);
    });
    
    // BFS search for augmenting paths
    function findAugmentingPath() {
        const parent = {};
        const visited = new Set();
        const queue = [s];
        visited.add(s);
        
        while (queue.length > 0) {
            const curr = queue.shift();
            if (curr === t) {
                // Reconstruct path
                const path = [];
                let node = t;
                while (node !== s) {
                    path.push(node);
                    node = parent[node];
                }
                path.push(s);
                return path.reverse();
            }
            
            const neighbors = residualAdj[curr] || [];
            for (let v of neighbors) {
                const residualCap = capacity[`${curr}->${v}`] - flow[`${curr}->${v}`];
                if (!visited.has(v) && residualCap > 0) {
                    parent[v] = curr;
                    visited.add(v);
                    queue.push(v);
                }
            }
        }
        return null;
    }
    
    // Edmonds-Karp Loop
    let path;
    while ((path = findAugmentingPath()) !== null) {
        // Augment flow along path
        for (let i = 0; i < path.length - 1; i++) {
            const u = path[i];
            const v = path[i+1];
            flow[`${u}->${v}`] += 1;
            flow[`${v}->${u}`] -= 1;
        }
    }
    
    // Run BFS from source in residual graph to find reachable component S (Class 0 partition)
    const reachable = new Set();
    const queue = [s];
    reachable.add(s);
    
    while (queue.length > 0) {
        const curr = queue.shift();
        const neighbors = residualAdj[curr] || [];
        for (let v of neighbors) {
            const residualCap = capacity[`${curr}->${v}`] - flow[`${curr}->${v}`];
            if (!reachable.has(v) && residualCap > 0) {
                reachable.add(v);
                queue.push(v);
            }
        }
    }
    
    // Assign hard binary predictions
    nodes.forEach(n => {
        if (reachable.has(n.id)) {
            n.predProb = 0.0; // Source side -> Class 0
        } else {
            n.predProb = 1.0; // Sink side -> Class 1
        }
    });
}

// C. GCN LP (Kipf & Welling Convolution Solver)
function solveGCNLP(labels) {
    const nodes = simulatorState.nodes;
    const N = nodes.length;
    const adj = simulatorState.adjList;
    
    // 1. Build adjacency with self-loops degree counts
    const degrees = new Array(N).fill(0);
    for (let i = 0; i < N; i++) {
        degrees[i] = adj[i].length + 1; // +1 for self-loop
    }
    
    // 2. Initialize indicator representations H^(0) of shape N x 2
    // Class 0 -> [1.0, 0.0]
    // Class 1 -> [0.0, 1.0]
    // Unlabeled -> [0.0, 0.0] (no initial features to allow scale-invariant diffusion)
    let H = [];
    for (let i = 0; i < N; i++) {
        if (i in labels) {
            if (labels[i] === 0) {
                H.push([1.0, 0.0]);
            } else {
                H.push([0.0, 1.0]);
            }
        } else {
            H.push([0.0, 0.0]);
        }
    }
    
    // 3. Compute dynamic geodesic coverage layers (L) based on seed set proximity
    const seedIndices = Object.keys(labels).map(Number);
    let n_layers = 2;
    if (seedIndices.length > 0) {
        const dist = new Array(N).fill(Infinity);
        const queue = [];
        for (let s of seedIndices) {
            dist[s] = 0;
            queue.push(s);
        }
        
        let maxDist = 0;
        let head = 0;
        while (head < queue.length) {
            const u = queue[head++];
            const d = dist[u];
            if (d > maxDist) {
                maxDist = d;
            }
            const neighbors = adj[u];
            if (neighbors) {
                for (let v of neighbors) {
                    if (dist[v] === Infinity) {
                        dist[v] = d + 1;
                        queue.push(v);
                    }
                }
            }
        }
        n_layers = maxDist;
    }
    if (n_layers === 0) {
        n_layers = 2;
    }
    
    if (simulatorState.lastGcnLayers !== n_layers) {
        simulatorState.lastGcnLayers = n_layers;
        addLog(`GCN dynamically scaled convolution depth to L = ${n_layers} layers (geodesic radius)`);
    }
    
    // 4. Run L layers of GCN convolutions
    for (let layer = 0; layer < n_layers; layer++) {
        let next_H = [];
        for (let i = 0; i < N; i++) {
            let sum0 = 0;
            let sum1 = 0;
            const d_i = degrees[i];
            
            // Self loop contribution
            const w_ii = 1.0 / d_i;
            sum0 += H[i][0] * w_ii;
            sum1 += H[i][1] * w_ii;
            
            // Neighbors contribution
            const neighbors = adj[i];
            if (neighbors) {
                for (let v of neighbors) {
                    const d_v = degrees[v];
                    const weight = 1.0 / Math.sqrt(d_i * d_v);
                    sum0 += H[v][0] * weight;
                    sum1 += H[v][1] * weight;
                }
            }
            next_H.push([sum0, sum1]);
        }
        H = next_H;
    }
    
    // 5. ApplySoftmax/Ratio normalisation and assign predictions to nodes
    for (let i = 0; i < N; i++) {
        const row = H[i];
        const row_sum = row[0] + row[1];
        if (row_sum > 0) {
            nodes[i].predProb = row[1] / row_sum;
        } else {
            nodes[i].predProb = 0.5;
        }
    }
}

// ---------------------------------------------------------
// 3. Active Learning Samplers
// ---------------------------------------------------------
function executeActiveStep() {
    if (simulatorState.nodes.length === 0) return;
    
    // 1. Find unlabeled set
    const unlabeled = simulatorState.nodes.filter(n => !n.isQueried);
    if (unlabeled.length === 0) {
        addLog("All nodes have been queried! Simulation complete.");
        stopAutoPlay();
        return;
    }
    
    // 2. Select node based on Sampler
    const samplerType = simulatorState.samplerType;
    let selectedNodeId = null;
    
    if (samplerType === 'random') {
        selectedNodeId = sampleRandom(unlabeled);
    } else if (samplerType === 's2') {
        selectedNodeId = sampleS2(unlabeled);
    } else if (samplerType === 'harmonic_greedy') {
        selectedNodeId = sampleHarmonicGreedy(unlabeled);
    } else if (samplerType === 'featprop') {
        selectedNodeId = sampleFeatProp(unlabeled);
    }
    
    if (selectedNodeId === null) {
        selectedNodeId = sampleRandom(unlabeled);
    }
    
    // 3. Query selected node
    const node = simulatorState.nodes[selectedNodeId];
    node.isQueried = true;
    simulatorState.labeledSet[node.id] = node.label;
    simulatorState.currentStep += 1;
    
    addLog(`[Step ${simulatorState.currentStep}] ${samplerType.toUpperCase()} queried Node ${node.id} (Ground Truth: ${node.label})`);
    
    // 4. Update predictions
    runPropagation();
    
    // 5. Calculate new expected risk
    recordCurrentRisk();
    
    // 6. Draw & Update
    drawGraph();
    updateUIControls();
}

// A. Random Sampler
function sampleRandom(unlabeled) {
    const idx = Math.floor(Math.random() * unlabeled.length);
    return unlabeled[idx].id;
}

// B. S2 Sampler (Shortest Path Boundary Bisection)
function sampleS2(unlabeled) {
    const labels = simulatorState.labeledSet;
    const adj = simulatorState.adjList;
    
    // S2 requires representatives from both classes
    const class0 = Object.keys(labels).filter(id => labels[id] === 0).map(Number);
    const class1 = Object.keys(labels).filter(id => labels[id] === 1).map(Number);
    
    if (class0.length === 0 || class1.length === 0) {
        addLog("S2 Fallback: Missing labels for one class. Sampling randomly.");
        return sampleRandom(unlabeled);
    }
    
    const labelLookup = { ...labels };
    
    // Perform BFS from all Class 0 nodes to find shortest path to any Class 1 node
    // in a label-pruned graph (edges between opposite labels are removed)
    const visited = new Set();
    const parent = {};
    const queue = [];
    
    class0.forEach(nodeId => {
        queue.push(nodeId);
        visited.add(nodeId);
    });
    
    let pathFound = null;
    
    while (queue.length > 0) {
        const curr = queue.shift();
        
        if (labelLookup[curr] === 1) {
            const path = [];
            let n = curr;
            while (n !== undefined) {
                path.push(n);
                n = parent[n];
            }
            pathFound = path.reverse();
            break;
        }
        
        const neighbors = adj[curr] || [];
        for (let v of neighbors) {
            // Prune edge if both endpoints are labeled with different values (S2 Edge Removal)
            const l_curr = labelLookup[curr];
            const l_v = labelLookup[v];
            if (l_curr !== undefined && l_v !== undefined && l_curr !== l_v) {
                continue; // Pruned edge
            }
            
            if (!visited.has(v)) {
                visited.add(v);
                parent[v] = curr;
                queue.push(v);
            }
        }
    }
    
    if (!pathFound || pathFound.length <= 2) {
        addLog("S2 Fallback: No connection path found between classes. Sampling randomly.");
        return sampleRandom(unlabeled);
    }
    
    const unlabeledOnPath = pathFound.filter(nodeId => !(nodeId in labels));
    if (unlabeledOnPath.length === 0) {
        return sampleRandom(unlabeled);
    }
    
    const midIdx = Math.floor(unlabeledOnPath.length / 2);
    const selected = unlabeledOnPath[midIdx];
    
    addLog(`S2 Sampler bisected shortest path of length ${pathFound.length}: Selected Node ${selected}`);
    return selected;
}

// C. Harmonic Greedy Sampler (Uncertainty expected reduction)
function sampleHarmonicGreedy(unlabeled) {
    const labels = simulatorState.labeledSet;
    const nodes = simulatorState.nodes;
    const adj = simulatorState.adjList;
    
    // Subsample candidate nodes for computational fluidity in the browser
    const numCandidates = Math.min(15, unlabeled.length);
    const candidates = [];
    const shuf = [...unlabeled].sort(() => 0.5 - Math.random());
    for (let i = 0; i < numCandidates; i++) {
        candidates.push(shuf[i]);
    }
    
    let bestNodeId = null;
    let minExpectedRisk = Infinity;
    
    function calcRisk(probsDict, currentLabels) {
        let riskSum = 0;
        nodes.forEach(n => {
            if (!(n.id in currentLabels)) {
                const p = probsDict[n.id];
                riskSum += Math.min(p, 1 - p);
            }
        });
        return riskSum;
    }
    
    function simulateHarmonicLP(simLabels) {
        let probs = {};
        nodes.forEach(n => {
            if (n.id in simLabels) {
                probs[n.id] = simLabels[n.id];
            } else {
                probs[n.id] = 0.5;
            }
        });
        
        for (let step = 0; step < 25; step++) {
            let nextProbs = { ...probs };
            nodes.forEach(n => {
                if (!(n.id in simLabels)) {
                    const neighbors = adj[n.id];
                    if (neighbors.length > 0) {
                        let sum = 0;
                        neighbors.forEach(v => { sum += probs[v]; });
                        nextProbs[n.id] = sum / neighbors.length;
                    }
                }
            });
            probs = nextProbs;
        }
        return probs;
    }
    
    candidates.forEach(cand => {
        const candId = cand.id;
        const p1 = cand.predProb;
        const p0 = 1 - p1;
        
        const labels1 = { ...labels, [candId]: 1 };
        const probs1 = simulateHarmonicLP(labels1);
        const risk1 = calcRisk(probs1, labels1);
        
        const labels0 = { ...labels, [candId]: 0 };
        const probs0 = simulateHarmonicLP(labels0);
        const risk0 = calcRisk(probs0, labels0);
        
        const expectedRisk = p1 * risk1 + p0 * risk0;
        if (expectedRisk < minExpectedRisk) {
            minExpectedRisk = expectedRisk;
            bestNodeId = candId;
        }
    });
    
    return bestNodeId;
}

// D. FeatProp Sampler (K-Medoids Clustering on Convolved Features)
function precomputeFeatPropFeatures() {
    const nodes = simulatorState.nodes;
    const N = nodes.length;
    if (N === 0) return;
    
    // Initialize node features: normalized [x, y] coordinates
    const X = [];
    for (let i = 0; i < N; i++) {
        const n = nodes[i];
        if (n.gridX !== undefined && n.gridY !== undefined) {
            // Grid coordinates
            X.push([n.gridX, n.gridY]);
        } else {
            // Normalized coordinates for bipartite/SBM
            X.push([n.x / CANVAS_WIDTH, n.y / CANVAS_HEIGHT]);
        }
    }
    
    // Degrees with self-loops
    const degrees = new Array(N);
    for (let i = 0; i < N; i++) {
        degrees[i] = simulatorState.adjList[i].length + 1;
    }
    
    // Convolve for 2 propagation steps (same as n_prop_steps in python class)
    let H = X.map(row => [...row]);
    const n_prop_steps = 2;
    for (let step = 0; step < n_prop_steps; step++) {
        let next_H = [];
        for (let i = 0; i < N; i++) {
            let sum0 = H[i][0] / degrees[i];
            let sum1 = H[i][1] / degrees[i];
            const neighbors = simulatorState.adjList[i] || [];
            for (let k = 0; k < neighbors.length; k++) {
                const v = neighbors[k];
                const weight = 1.0 / Math.sqrt(degrees[i] * degrees[v]);
                sum0 += H[v][0] * weight;
                sum1 += H[v][1] * weight;
            }
            next_H.push([sum0, sum1]);
        }
        H = next_H;
    }
    
    simulatorState.featPropFeatures = H;
    addLog(`Precomputed FeatProp convolved feature representations (${n_prop_steps} layers)`);
}

function sampleFeatProp(unlabeled) {
    const labels = simulatorState.labeledSet;
    const X_bar = simulatorState.featPropFeatures;
    const N = simulatorState.nodes.length;
    
    if (!X_bar || X_bar.length !== N) {
        addLog("FeatProp Error: Features not precomputed. Sampling randomly.");
        return sampleRandom(unlabeled);
    }
    
    const labeled_ids = Object.keys(labels).map(Number);
    
    // Euclidean distance in convolved space
    function dist(i, j) {
        const dx = X_bar[i][0] - X_bar[j][0];
        const dy = X_bar[i][1] - X_bar[j][1];
        return Math.sqrt(dx*dx + dy*dy);
    }
    
    // If no labeled nodes yet, pick the medoid of the entire convolved feature space
    if (labeled_ids.length === 0) {
        let min_sum = Infinity;
        let medoid_idx = 0;
        for (let i = 0; i < N; i++) {
            let sum = 0;
            for (let j = 0; j < N; j++) {
                sum += dist(i, j);
            }
            if (sum < min_sum) {
                min_sum = sum;
                medoid_idx = i;
            }
        }
        addLog(`FeatProp selected initial medoid Node ${medoid_idx}`);
        return medoid_idx;
    }
    
    // Compute distance from all nodes to closest labeled node
    const min_dists = new Array(N);
    for (let i = 0; i < N; i++) {
        let m_d = Infinity;
        for (let k = 0; k < labeled_ids.length; k++) {
            const d = dist(i, labeled_ids[k]);
            if (d < m_d) {
                m_d = d;
            }
        }
        min_dists[i] = m_d;
    }
    
    let best_candidate = null;
    let min_total_cost = Infinity;
    const candidates = unlabeled.map(n => n.id);
    
    // Find candidate that minimizes total K-Medoids objective
    for (let c = 0; c < candidates.length; c++) {
        const cand = candidates[c];
        let total_cost = 0;
        for (let i = 0; i < N; i++) {
            const d_cand = dist(i, cand);
            total_cost += Math.min(min_dists[i], d_cand);
        }
        if (total_cost < min_total_cost) {
            min_total_cost = total_cost;
            best_candidate = cand;
        }
    }
    
    return best_candidate;
}

// ---------------------------------------------------------
// 4. HTML5 Canvas Renderer
// ---------------------------------------------------------
function drawGraph() {
    if (!ctx) return;
    
    // Clear Canvas
    ctx.fillStyle = '#0f172a'; // slate-900 background
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    
    const nodes = simulatorState.nodes;
    const edges = simulatorState.edges;
    
    // 1. Draw Edges (Colored to stand out beautifully)
    edges.forEach(e => {
        const uNode = nodes[e.u];
        const vNode = nodes[e.v];
        
        // S2 Edge Removal Pruning check: connects queried nodes with conflicting labels
        const u_curr = simulatorState.labeledSet[e.u];
        const v_curr = simulatorState.labeledSet[e.v];
        const isPruned = (u_curr !== undefined && v_curr !== undefined && u_curr !== v_curr);
        
        ctx.beginPath();
        ctx.moveTo(uNode.x, uNode.y);
        ctx.lineTo(vNode.x, vNode.y);
        
        if (isPruned) {
            // Pruned edge under S2: stands out as dashed neon red to show the edge removal visually!
            ctx.strokeStyle = 'rgba(239, 68, 68, 0.85)'; // Neon Red
            ctx.lineWidth = 1.8;
            ctx.setLineDash([3, 4]); // Clean dash strokes
        } else {
            // Active edges: Color based on prediction values so they stand out dynamically
            const pU = uNode.predProb;
            const pV = vNode.predProb;
            
            if (pU < 0.35 && pV < 0.35) {
                // Class 0 (Blue) dominant community edge
                ctx.strokeStyle = 'rgba(37, 99, 235, 0.5)';
                ctx.lineWidth = 1.3;
            } else if (pU > 0.65 && pV > 0.65) {
                // Class 1 (Red) dominant community edge
                ctx.strokeStyle = 'rgba(225, 29, 72, 0.5)';
                ctx.lineWidth = 1.3;
            } else {
                // Boundary / Uncertain / Connecting different components
                ctx.strokeStyle = 'rgba(148, 163, 184, 0.45)'; // Bright silver/slate
                ctx.lineWidth = 1.1;
            }
            ctx.setLineDash([]); // Reset to solid lines
        }
        ctx.stroke();
    });
    ctx.setLineDash([]); // Reset line dash for nodes
    
    // 2. Draw Nodes
    const type = simulatorState.graphType;
    const rNode = (type === 'clustered') ? 10 : 6;
    
    nodes.forEach(n => {
        ctx.beginPath();
        ctx.arc(n.x, n.y, rNode, 0, Math.PI * 2);
        
        let color = '#64748b';
        let border = '#475569';
        let borderWidth = 1.0;
        
        if (n.isQueried) {
            // Queried nodes: Solid blue/red with dense, thick border
            color = (n.label === 0) ? '#2563eb' : '#e11d48';
            border = (n.label === 0) ? '#1d4ed8' : '#be123c';
            borderWidth = 3.5;
        } else {
            // Estimated nodes: Smooth color gradients based on probabilities
            const p = n.predProb;
            color = getRGBColor(p);
            border = 'rgba(255, 255, 255, 0.15)';
            borderWidth = 1.0;
        }
        
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = border;
        ctx.lineWidth = borderWidth;
        ctx.stroke();
        
        // Special ring for hovered node
        if (simulatorState.hoveredNode === n.id) {
            ctx.beginPath();
            ctx.arc(n.x, n.y, rNode + 5, 0, Math.PI * 2);
            ctx.strokeStyle = '#f59e0b';
            ctx.lineWidth = 2.0;
            ctx.stroke();
        }
    });
}

function getRGBColor(p) {
    // Interpolate beautifully: Blue (37, 99, 235) -> Neutral Grey (51, 65, 85) -> Red (225, 29, 72)
    let r, g, b;
    if (p < 0.5) {
        const t = p / 0.5;
        r = Math.round(37 + (51 - 37) * t);
        g = Math.round(99 + (65 - 99) * t);
        b = Math.round(235 + (85 - 235) * t);
    } else {
        const t = (p - 0.5) / 0.5;
        r = Math.round(51 + (225 - 51) * t);
        g = Math.round(65 + (29 - 65) * t);
        b = Math.round(85 + (72 - 85) * t);
    }
    return `rgb(${r}, ${g}, ${b})`;
}

// ---------------------------------------------------------
// 5. Simulation Event Handlers & UI Updates
// ---------------------------------------------------------
function handleCanvasClick(e) {
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    let closestNode = null;
    let minDist = Infinity;
    const type = simulatorState.graphType;
    const clickLimit = (type === 'clustered') ? 20 : 12;
    
    simulatorState.nodes.forEach(n => {
        const dx = n.x - mouseX;
        const dy = n.y - mouseY;
        const dist = Math.sqrt(dx*dx + dy*dy);
        if (dist < clickLimit && dist < minDist) {
            minDist = dist;
            closestNode = n;
        }
    });
    
    if (closestNode && !closestNode.isQueried) {
        closestNode.isQueried = true;
        simulatorState.labeledSet[closestNode.id] = closestNode.label;
        simulatorState.currentStep += 1;
        
        addLog(`[Step ${simulatorState.currentStep}] Manual User Query on Node ${closestNode.id} (Ground Truth: ${closestNode.label})`);
        
        runPropagation();
        recordCurrentRisk();
        drawGraph();
        updateUIControls();
    }
}

function handleCanvasMouseMove(e) {
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    let closestNode = null;
    let minDist = Infinity;
    const type = simulatorState.graphType;
    const hoverLimit = (type === 'clustered') ? 15 : 10;
    
    simulatorState.nodes.forEach(n => {
        const dx = n.x - mouseX;
        const dy = n.y - mouseY;
        const dist = Math.sqrt(dx*dx + dy*dy);
        if (dist < hoverLimit && dist < minDist) {
            minDist = dist;
            closestNode = n;
        }
    });
    
    const prevHovered = simulatorState.hoveredNode;
    if (closestNode) {
        simulatorState.hoveredNode = closestNode.id;
        updateHoverTooltip(closestNode);
    } else {
        simulatorState.hoveredNode = null;
        clearHoverTooltip();
    }
    
    if (prevHovered !== simulatorState.hoveredNode) {
        drawGraph();
    }
}

function updateHoverTooltip(node) {
    const panel = document.getElementById('hover-tooltip-panel');
    if (panel) {
        panel.innerHTML = `
            <div class="tip-row"><strong>Node ID:</strong> <span>${node.id}</span></div>
            <div class="tip-row"><strong>Actual Label:</strong> <span class="label-badge ${node.label === 0 ? 'l-0' : 'l-1'}">${node.label}</span></div>
            <div class="tip-row"><strong>Status:</strong> <span>${node.isQueried ? '<strong class="gold-text">Queried</strong>' : 'Estimated'}</span></div>
            <div class="tip-row"><strong>Estimated Prob:</strong> <span>${(node.predProb * 100).toFixed(1)}% (Class 1)</span></div>
        `;
    }
}

function clearHoverTooltip() {
    const panel = document.getElementById('hover-tooltip-panel');
    if (panel) {
        panel.innerHTML = `<span class="text-muted italic">Hover over a node to inspect...</span>`;
    }
}

function calculatePredictionAccuracy() {
    let correct = 0;
    simulatorState.nodes.forEach(n => {
        const predClass = n.predProb >= 0.5 ? 1 : 0;
        if (predClass === n.label) {
            correct++;
        }
    });
    return (correct / simulatorState.nodes.length) * 100;
}

function recordCurrentRisk() {
    let currentRisk = 0;
    simulatorState.nodes.forEach(n => {
        if (!(n.id in simulatorState.labeledSet)) {
            const p = n.predProb;
            currentRisk += Math.min(p, 1 - p);
        }
    });
    
    simulatorState.history.push({
        step: simulatorState.currentStep,
        risk: currentRisk
    });
    
    const riskValEl = document.getElementById('gauge-risk-value');
    const riskFillEl = document.getElementById('gauge-risk-fill');
    
    if (riskValEl && riskFillEl) {
        riskValEl.textContent = currentRisk.toFixed(2);
        const maxRisk = (simulatorState.graphType === 'clustered') ? 20 : 200;
        const pct = Math.min(100, Math.max(0, (currentRisk / maxRisk) * 100));
        riskFillEl.style.width = `${pct}%`;
        
        if (pct < 15) {
            riskFillEl.style.backgroundColor = '#10b981';
        } else if (pct < 50) {
            riskFillEl.style.backgroundColor = '#f59e0b';
        } else {
            riskFillEl.style.backgroundColor = '#e11d48';
        }
    }
}

function addLog(msg) {
    simulatorState.logFeed.unshift(msg);
    if (simulatorState.logFeed.length > 25) {
        simulatorState.logFeed.pop();
    }
    
    const logEl = document.getElementById('simulator-log-feed');
    if (logEl) {
        logEl.innerHTML = simulatorState.logFeed.map(log => {
            if (log.includes('queried')) {
                return `<div class="log-line log-query">${log}</div>`;
            } else if (log.includes('Initializing')) {
                return `<div class="log-line log-init">${log}</div>`;
            }
            return `<div class="log-line">${log}</div>`;
        }).join('');
    }
}

function updateUIControls() {
    const stepEl = document.getElementById('sim-current-step');
    const totalEl = document.getElementById('sim-queried-total');
    const accuracyEl = document.getElementById('sim-accuracy');
    
    if (stepEl) stepEl.textContent = simulatorState.currentStep;
    if (totalEl) {
        const total = Object.keys(simulatorState.labeledSet).length;
        totalEl.textContent = `${total} / ${simulatorState.nodes.length}`;
    }
    if (accuracyEl) {
        const acc = calculatePredictionAccuracy();
        accuracyEl.textContent = `${acc.toFixed(1)}%`;
    }
}

// ---------------------------------------------------------
// 6. Interactive Controls Interface
// ---------------------------------------------------------
function changeGraphType(val) {
    simulatorState.graphType = val;
    initializeGraph();
}

function changeSamplerType(val) {
    simulatorState.samplerType = val;
    addLog(`Sampler switched to: ${val.toUpperCase()}`);
    initializeGraph();
}

function changePropagatorType(val) {
    simulatorState.propagatorType = val;
    addLog(`Propagator switched to: ${val.toUpperCase()}`);
    runPropagation();
    recordCurrentRisk();
    drawGraph();
    updateUIControls();
}

function toggleAutoPlay() {
    if (simulatorState.isPlaying) {
        stopAutoPlay();
    } else {
        startAutoPlay();
    }
}

function startAutoPlay() {
    simulatorState.isPlaying = true;
    const playBtn = document.getElementById('btn-play-pause');
    if (playBtn) playBtn.innerHTML = '❚❚ Pause';
    
    simulatorState.playInterval = setInterval(() => {
        executeActiveStep();
    }, 550);
}

function stopAutoPlay() {
    simulatorState.isPlaying = false;
    const playBtn = document.getElementById('btn-play-pause');
    if (playBtn) playBtn.innerHTML = '▶ Auto Play';
    
    if (simulatorState.playInterval) {
        clearInterval(simulatorState.playInterval);
        simulatorState.playInterval = null;
    }
}

// ==========================================================================
// 7. Card Demo Visualizations (Harmonic, Min-Cut, Spectral)
// ==========================================================================

function initializeCardDemos() {
    initHarmonicCardDemo();
    initMinCutCardDemo();
    initSpectralCardDemo();
}

// A. Harmonic Card Demo
let harmonicCard = {
    canvas: null,
    ctx: null,
    nodes: [],
    values: [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0],
    initialValues: [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0],
    iter: 0,
    isPlaying: false,
    interval: null
};

function initHarmonicCardDemo() {
    harmonicCard.canvas = document.getElementById('canvas-harmonic');
    if (!harmonicCard.canvas) return;
    harmonicCard.ctx = harmonicCard.canvas.getContext('2d');
    
    // Set up nodes
    const nodeCount = 7;
    const spacing = 450 / (nodeCount - 1);
    const startX = 50;
    const y = 80;
    
    harmonicCard.nodes = [];
    for (let i = 0; i < nodeCount; i++) {
        harmonicCard.nodes.push({
            id: i,
            x: startX + i * spacing,
            y: y,
            isSeed: (i === 0 || i === nodeCount - 1),
            label: (i === 0) ? 0 : ((i === nodeCount - 1) ? 1 : null)
        });
    }
    
    // Set up controls
    const playBtn = document.getElementById('btn-harmonic-play');
    const stepBtn = document.getElementById('btn-harmonic-step');
    const resetBtn = document.getElementById('btn-harmonic-reset');
    
    if (playBtn) playBtn.onclick = toggleHarmonicPlay;
    if (stepBtn) stepBtn.onclick = stepHarmonicDemo;
    if (resetBtn) resetBtn.onclick = resetHarmonicDemo;
    
    resetHarmonicDemo();
}

function drawHarmonicDemo() {
    const ctx = harmonicCard.ctx;
    if (!ctx) return;
    
    ctx.fillStyle = '#0f172a'; // slate-900
    ctx.fillRect(0, 0, 550, 340);
    
    // Draw connections between nodes
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)';
    ctx.lineWidth = 4;
    ctx.moveTo(harmonicCard.nodes[0].x, harmonicCard.nodes[0].y);
    for (let i = 1; i < harmonicCard.nodes.length; i++) {
        ctx.lineTo(harmonicCard.nodes[i].x, harmonicCard.nodes[i].y);
    }
    ctx.stroke();
    
    // Draw nodes
    harmonicCard.nodes.forEach(n => {
        const val = harmonicCard.values[n.id];
        ctx.beginPath();
        ctx.arc(n.x, n.y, 22, 0, Math.PI * 2);
        
        ctx.fillStyle = getRGBColor(val);
        ctx.fill();
        
        if (n.isSeed) {
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 3;
        } else {
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
            ctx.lineWidth = 1.5;
        }
        ctx.stroke();
        
        // Draw value text inside/above node
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 11px Inter';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(val.toFixed(2), n.x, n.y);
        
        // Draw seed badge
        if (n.isSeed) {
            ctx.fillStyle = '#f59e0b';
            ctx.font = 'bold 9px Outfit';
            ctx.fillText("SEED", n.x, n.y - 32);
        }
    });
    
    // Draw line plot mapping values
    const chartYStart = 160;
    const chartYEnd = 290;
    const chartHeight = chartYEnd - chartYStart;
    
    // Draw chart axes
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.3)';
    ctx.lineWidth = 1.5;
    // Y Axis
    ctx.moveTo(50, chartYStart - 10);
    ctx.lineTo(50, chartYEnd);
    // X Axis
    ctx.lineTo(500, chartYEnd);
    ctx.stroke();
    
    // Draw axis labels
    ctx.fillStyle = '#64748b';
    ctx.font = '10px Inter';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText("1.0", 40, chartYStart);
    ctx.fillText("0.5", 40, chartYStart + chartHeight/2);
    ctx.fillText("0.0", 40, chartYEnd);
    
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText("Node Index", 275, chartYEnd + 15);
    for (let i = 0; i < harmonicCard.nodes.length; i++) {
        ctx.fillText(i, harmonicCard.nodes[i].x, chartYEnd + 5);
    }
    
    // Draw grid lines
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.08)';
    ctx.lineWidth = 1;
    ctx.moveTo(50, chartYStart + chartHeight/2);
    ctx.lineTo(500, chartYStart + chartHeight/2);
    ctx.moveTo(50, chartYStart);
    ctx.lineTo(500, chartYStart);
    ctx.stroke();
    
    // Draw data points and line on the chart
    ctx.beginPath();
    ctx.strokeStyle = '#8b5cf6'; // Violet line
    ctx.lineWidth = 3;
    let py = chartYEnd - harmonicCard.values[0] * chartHeight;
    ctx.moveTo(harmonicCard.nodes[0].x, py);
    for (let i = 1; i < harmonicCard.nodes.length; i++) {
        py = chartYEnd - harmonicCard.values[i] * chartHeight;
        ctx.lineTo(harmonicCard.nodes[i].x, py);
    }
    ctx.stroke();
    
    // Draw dots on data points
    for (let i = 0; i < harmonicCard.nodes.length; i++) {
        py = chartYEnd - harmonicCard.values[i] * chartHeight;
        ctx.beginPath();
        ctx.arc(harmonicCard.nodes[i].x, py, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#c084fc';
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }
    
    // Update iteration label
    const info = document.getElementById('info-harmonic');
    if (info) info.textContent = `Iter: ${harmonicCard.iter}`;
}

function stepHarmonicDemo() {
    const nextVals = [...harmonicCard.values];
    const nodeCount = harmonicCard.nodes.length;
    
    // Jacobi iteration formula
    for (let i = 1; i < nodeCount - 1; i++) {
        nextVals[i] = (harmonicCard.values[i - 1] + harmonicCard.values[i + 1]) / 2;
    }
    
    harmonicCard.values = nextVals;
    harmonicCard.iter += 1;
    drawHarmonicDemo();
}

function toggleHarmonicPlay() {
    const playBtn = document.getElementById('btn-harmonic-play');
    if (harmonicCard.isPlaying) {
        clearInterval(harmonicCard.interval);
        harmonicCard.isPlaying = false;
        if (playBtn) playBtn.textContent = '▶ Play';
    } else {
        harmonicCard.isPlaying = true;
        if (playBtn) playBtn.textContent = '❚❚ Pause';
        harmonicCard.interval = setInterval(() => {
            stepHarmonicDemo();
            let maxDiff = 0;
            for (let i = 1; i < harmonicCard.nodes.length - 1; i++) {
                const ideal = i / (harmonicCard.nodes.length - 1);
                maxDiff = Math.max(maxDiff, Math.abs(harmonicCard.values[i] - ideal));
            }
            if (maxDiff < 0.005 || harmonicCard.iter > 60) {
                toggleHarmonicPlay();
            }
        }, 350);
    }
}

function resetHarmonicDemo() {
    if (harmonicCard.isPlaying) {
        toggleHarmonicPlay();
    }
    harmonicCard.values = [...harmonicCard.initialValues];
    harmonicCard.iter = 0;
    drawHarmonicDemo();
}

// B. Min-Cut Card Demo
let mincutCard = {
    canvas: null,
    ctx: null,
    nodes: [],
    edges: [],
    reachable: new Set(),
    nonReachable: new Set(),
    cutEdges: [],
    step: 0,
    playInterval: null,
    isPlaying: false
};

function initMinCutCardDemo() {
    mincutCard.canvas = document.getElementById('canvas-mincut');
    if (!mincutCard.canvas) return;
    mincutCard.ctx = mincutCard.canvas.getContext('2d');
    
    const coords = [
        [90, 110], [75, 160], [100, 210], [130, 230], [160, 200],
        [170, 150], [140, 100], [110, 150], [135, 160], [105, 120],
        [450, 110], [470, 160], [440, 210], [410, 230], [380, 200],
        [370, 150], [400, 100], [430, 150], [405, 160], [435, 120],
        [240, 130], [230, 190], [250, 230], [300, 120], [310, 180],
        [295, 230], [270, 150], [280, 210], [255, 170], [285, 160]
    ];
    
    mincutCard.nodes = coords.map((c, i) => {
        return {
            id: i,
            x: c[0],
            y: c[1] - 10,
            label: (i < 10) ? 0 : ((i < 20) ? 1 : null),
            isSeed: (i < 20)
        };
    });
    
    const edgeList = [
        [0, 1], [0, 6], [0, 9], [1, 2], [1, 7], [1, 8], [2, 3], [2, 7], [3, 4], [3, 8],
        [4, 5], [4, 8], [5, 6], [5, 8], [6, 9], [7, 8], [7, 9], [0, 7], [2, 4], [1, 5],
        [10, 11], [10, 16], [10, 19], [11, 12], [11, 17], [11, 18], [12, 13], [12, 17], [13, 14], [13, 18],
        [14, 15], [14, 18], [15, 16], [15, 18], [16, 19], [17, 18], [17, 19], [10, 17], [12, 14], [11, 15],
        [20, 21], [20, 26], [21, 22], [21, 27], [22, 25], [23, 24], [23, 29], [24, 25], [24, 29],
        [26, 27], [26, 28], [27, 28], [20, 28], [23, 26], [21, 28], [24, 27],
        [5, 20], [4, 21], [3, 22], [8, 28],
        [23, 15], [24, 14], [25, 13], [29, 18]
    ];
    mincutCard.edges = edgeList.map(e => ({ u: e[0], v: e[1] }));
    
    mincutCard.reachable = new Set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 26, 27, 28]);
    mincutCard.nonReachable = new Set([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 29]);
    
    mincutCard.cutEdges = mincutCard.edges.filter(e => {
        const uReachable = mincutCard.reachable.has(e.u);
        const vReachable = mincutCard.reachable.has(e.v);
        return (uReachable && !vReachable) || (!uReachable && vReachable);
    });
    
    const playBtn = document.getElementById('btn-mincut-play');
    const resetBtn = document.getElementById('btn-mincut-reset');
    
    if (playBtn) playBtn.onclick = toggleMinCutPlay;
    if (resetBtn) resetBtn.onclick = resetMinCutDemo;
    
    resetMinCutDemo();
}

function drawMinCutDemo() {
    const ctx = mincutCard.ctx;
    if (!ctx) return;
    
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, 550, 340);
    
    const step = mincutCard.step;
    
    const srcX = 35;
    const srcY = 160;
    const snkX = 515;
    const snkY = 160;
    
    if (step >= 1) {
        ctx.fillStyle = 'rgba(16, 185, 129, 0.15)';
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 2.5;
        ctx.fillRect(srcX - 12, srcY - 12, 24, 24);
        ctx.strokeRect(srcX - 12, srcY - 12, 24, 24);
        
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 11px Outfit';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText("s", srcX, srcY);
        ctx.fillStyle = '#10b981';
        ctx.font = 'bold 8px Outfit';
        ctx.fillText("SOURCE", srcX, srcY - 20);
        
        ctx.fillStyle = 'rgba(139, 92, 246, 0.15)';
        ctx.strokeStyle = '#8b5cf6';
        ctx.fillRect(snkX - 12, snkY - 12, 24, 24);
        ctx.strokeRect(snkX - 12, snkY - 12, 24, 24);
        
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 11px Outfit';
        ctx.fillText("t", snkX, snkY);
        ctx.fillStyle = '#8b5cf6';
        ctx.font = 'bold 8px Outfit';
        ctx.fillText("SINK", snkX, snkY - 20);
        
        ctx.lineWidth = 1.0;
        ctx.setLineDash([2, 3]);
        
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.3)';
        mincutCard.nodes.slice(0, 10).forEach(n => {
            ctx.beginPath();
            ctx.moveTo(srcX, srcY);
            ctx.lineTo(n.x, n.y);
            ctx.stroke();
        });
        
        ctx.strokeStyle = 'rgba(139, 92, 246, 0.3)';
        mincutCard.nodes.slice(10, 20).forEach(n => {
            ctx.beginPath();
            ctx.moveTo(n.x, n.y);
            ctx.lineTo(snkX, snkY);
            ctx.stroke();
        });
        
        ctx.setLineDash([]);
    }
    
    const augmentingEdges = new Set([
        '5->20', '20->26', '26->27', '27->28', '28->8',
        '23->15', '24->14', '25->13', '29->18'
    ]);
    
    mincutCard.edges.forEach(e => {
        const uNode = mincutCard.nodes[e.u];
        const vNode = mincutCard.nodes[e.v];
        
        ctx.beginPath();
        ctx.moveTo(uNode.x, uNode.y);
        ctx.lineTo(vNode.x, vNode.y);
        
        const isCutEdge = mincutCard.cutEdges.includes(e);
        
        if (step >= 3 && isCutEdge) {
            ctx.strokeStyle = 'rgba(245, 158, 11, 0.85)';
            ctx.lineWidth = 2.0;
            ctx.setLineDash([3, 4]);
        } else if (step === 2 && (augmentingEdges.has(`${e.u}->${e.v}`) || augmentingEdges.has(`${e.v}->${e.u}`))) {
            ctx.strokeStyle = '#f59e0b';
            ctx.lineWidth = 2.5;
            ctx.setLineDash([]);
        } else {
            ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)';
            ctx.lineWidth = 1.2;
            ctx.setLineDash([]);
        }
        ctx.stroke();
    });
    ctx.setLineDash([]);
    
    mincutCard.nodes.forEach(n => {
        ctx.beginPath();
        ctx.arc(n.x, n.y, 8, 0, Math.PI * 2);
        
        let color = '#475569';
        let border = 'rgba(255, 255, 255, 0.15)';
        let borderWidth = 1.0;
        
        if (step >= 3) {
            const isReachable = mincutCard.reachable.has(n.id);
            color = isReachable ? '#2563eb' : '#e11d48';
            border = isReachable ? '#1d4ed8' : '#be123c';
            borderWidth = n.isSeed ? 3.0 : 1.0;
        } else {
            if (n.isSeed) {
                color = (n.label === 0) ? '#2563eb' : '#e11d48';
                border = '#ffffff';
                borderWidth = 2.0;
            }
        }
        
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = border;
        ctx.lineWidth = borderWidth;
        ctx.stroke();
    });
    
    if (step >= 3) {
        ctx.beginPath();
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2.5;
        ctx.setLineDash([6, 6]);
        ctx.moveTo(275, 40);
        ctx.lineTo(275, 280);
        ctx.stroke();
        ctx.setLineDash([]);
        
        ctx.fillStyle = '#f59e0b';
        ctx.font = 'bold 9px Outfit';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText("MIN-CUT WALL boundary", 275, 35);
    }
    
    const info = document.getElementById('info-mincut');
    if (info) {
        if (step === 0) info.textContent = "Seeds Placed (20 Nodes)";
        else if (step === 1) info.textContent = "s-t Network Built";
        else if (step === 2) info.textContent = "Augmenting Paths Found";
        else if (step === 3) info.textContent = "Cut Computed & Partitioned";
    }
}

function stepMinCutDemo() {
    mincutCard.step = (mincutCard.step + 1) % 4;
    drawMinCutDemo();
}

function toggleMinCutPlay() {
    const playBtn = document.getElementById('btn-mincut-play');
    if (mincutCard.isPlaying) {
        clearInterval(mincutCard.playInterval);
        mincutCard.isPlaying = false;
        if (playBtn) playBtn.textContent = '▶ Animate Cut';
    } else {
        mincutCard.isPlaying = true;
        if (playBtn) playBtn.textContent = '❚❚ Pause';
        mincutCard.playInterval = setInterval(() => {
            stepMinCutDemo();
        }, 1500);
    }
}

function resetMinCutDemo() {
    if (mincutCard.isPlaying) {
        toggleMinCutPlay();
    }
    mincutCard.step = 0;
    drawMinCutDemo();
}

// C. Spectral Card Demo
let spectralCard = {
    canvas: null,
    ctx: null,
    nodes: [],
    edges: [],
    space: 'physical',
    transitionT: 0.0,
    animating: false
};

function initSpectralCardDemo() {
    spectralCard.canvas = document.getElementById('canvas-spectral');
    if (!spectralCard.canvas) return;
    spectralCard.ctx = spectralCard.canvas.getContext('2d');
    
    const physCoords = [
        [210, 110], [170, 130], [240, 160], [180, 190], [200, 240],
        [270, 130], [260, 200], [285, 230], [225, 175], [195, 150],
        
        [260, 100], [330, 130], [290, 170], [360, 190], [320, 240],
        [370, 140], [305, 220], [345, 225], [315, 150], [350, 110]
    ];
    
    const specCoords = [
        [100, 110], [115, 150], [140, 185], [160, 220], [120, 250],
        [180, 120], [190, 165], [175, 240], [145, 140], [150, 205],
        
        [380, 115], [395, 155], [420, 190], [440, 225], [400, 255],
        [460, 125], [470, 170], [455, 245], [425, 145], [430, 210]
    ];
    
    spectralCard.nodes = physCoords.map((c, i) => {
        return {
            id: i,
            physX: c[0],
            physY: c[1] - 10,
            specX: specCoords[i][0] + 25,
            specY: specCoords[i][1] - 10,
            drawX: c[0],
            drawY: c[1] - 10,
            label: (i < 10) ? 0 : 1,
            isSeed: (i === 0 || i === 19)
        };
    });
    
    const edgeList = [
        [0, 1], [0, 5], [1, 2], [1, 8], [2, 3], [2, 6], [3, 4], [3, 9], [4, 7], [5, 9], [6, 8], [7, 8], [0, 9],
        [10, 11], [10, 15], [11, 12], [11, 18], [12, 13], [12, 16], [13, 14], [13, 19], [14, 17], [15, 19], [16, 18], [17, 18], [10, 19],
        [2, 12], [6, 16], [5, 10], [8, 18]
    ];
    spectralCard.edges = edgeList.map(e => ({ u: e[0], v: e[1] }));
    
    const toggleBtn = document.getElementById('btn-spectral-toggle');
    if (toggleBtn) toggleBtn.onclick = toggleSpectralSpace;
    
    drawSpectralDemo();
}

function drawSpectralDemo() {
    const ctx = spectralCard.ctx;
    if (!ctx) return;
    
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, 550, 340);
    
    const t = spectralCard.transitionT;
    if (t > 0) {
        ctx.beginPath();
        ctx.strokeStyle = `rgba(245, 158, 11, ${t})`;
        ctx.lineWidth = 2.0;
        ctx.setLineDash([5, 5]);
        ctx.moveTo(275, 40);
        ctx.lineTo(275, 280);
        ctx.stroke();
        ctx.setLineDash([]);
        
        ctx.fillStyle = `rgba(245, 158, 11, ${t})`;
        ctx.font = 'bold 9px Outfit';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText("SVM DECISION BOUNDARY", 275, 35);
    }
    
    spectralCard.edges.forEach(e => {
        const uNode = spectralCard.nodes[e.u];
        const vNode = spectralCard.nodes[e.v];
        
        ctx.beginPath();
        ctx.moveTo(uNode.drawX, uNode.drawY);
        ctx.lineTo(vNode.drawX, vNode.drawY);
        
        const isCrossEdge = (e.u < 10 && e.v >= 10) || (e.v < 10 && e.u >= 10);
        if (isCrossEdge) {
            ctx.strokeStyle = `rgba(148, 163, 184, ${0.35 - t * 0.25})`;
            ctx.lineWidth = 1.0;
        } else {
            const label = (e.u < 10) ? 0 : 1;
            if (label === 0) {
                ctx.strokeStyle = `rgba(37, 99, 235, ${0.4 + t * 0.2})`;
            } else {
                ctx.strokeStyle = `rgba(225, 29, 72, ${0.4 + t * 0.2})`;
            }
            ctx.lineWidth = 1.2;
        }
        ctx.stroke();
    });
    
    spectralCard.nodes.forEach(n => {
        ctx.beginPath();
        ctx.arc(n.drawX, n.drawY, 8, 0, Math.PI * 2);
        
        let color = (n.label === 0) ? '#2563eb' : '#e11d48';
        let border = 'rgba(255, 255, 255, 0.15)';
        let borderWidth = 1.0;
        
        if (n.isSeed) {
            border = '#ffffff';
            borderWidth = 2.0;
        }
        
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = border;
        ctx.lineWidth = borderWidth;
        ctx.stroke();
    });
}

function toggleSpectralSpace() {
    if (spectralCard.animating) return;
    
    const targetSpace = (spectralCard.space === 'physical') ? 'spectral' : 'physical';
    spectralCard.space = targetSpace;
    spectralCard.animating = true;
    
    const duration = 850;
    const start = performance.now();
    
    const startT = spectralCard.transitionT;
    const endT = (targetSpace === 'spectral') ? 1.0 : 0.0;
    
    const info = document.getElementById('info-spectral');
    if (info) info.textContent = (targetSpace === 'spectral') ? "Spectral Space (v2 vs v3)" : "Physical Space";
    
    function animate(now) {
        const elapsed = now - start;
        const progress = Math.min(1.0, elapsed / duration);
        
        const ease = progress < 0.5 
            ? 4 * progress * progress * progress 
            : 1 - Math.pow(-2 * progress + 2, 3) / 2;
            
        spectralCard.transitionT = startT + (endT - startT) * ease;
        
        spectralCard.nodes.forEach(n => {
            n.drawX = n.physX + (n.specX - n.physX) * spectralCard.transitionT;
            n.drawY = n.physY + (n.specY - n.physY) * spectralCard.transitionT;
        });
        
        drawSpectralDemo();
        
        if (progress < 1.0) {
            requestAnimationFrame(animate);
        } else {
            spectralCard.animating = false;
        }
    }
    
    requestAnimationFrame(animate);
}
