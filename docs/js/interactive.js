/**
 * GraphALP Live Interactive Simulator
 * Pure vanilla JavaScript implementations of graph algorithms, samplers, and canvas visualizers.
 */

// Global State
let simulatorState = {
    graphType: 'grid_cols',          // 'grid_cols', 'grid_rows', 'grid_circ1', 'grid_circ2', 'bipartite'
    samplerType: 'harmonic_greedy',   // 'random', 's2', 'harmonic_greedy'
    propagatorType: 'harmonic',      // 'harmonic', 'mincut' (spectral removed)
    
    nodes: [],                       // List of node objects: { id, x, y, label, isQueried, predProb }
    edges: [],                       // Adjacency list: array of { u, v }
    adjList: {},                     // Node ID -> list of neighbor IDs
    labeledSet: {},                  // Node ID -> label (0 or 1)
    
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
    addLog(`Initializing graph: ${type.toUpperCase()}`);
    
    if (type.startsWith('grid_')) {
        generateGridGraph(type);
    } else if (type === 'bipartite') {
        generateBipartiteGraph();
    }
    
    // Build Adjacency List
    for (let i = 0; i < simulatorState.nodes.length; i++) {
        simulatorState.adjList[i] = [];
    }
    simulatorState.edges.forEach(e => {
        simulatorState.adjList[e.u].push(e.v);
        simulatorState.adjList[e.v].push(e.u);
    });
    
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

function generateBipartiteGraph() {
    // 40 nodes: left partition 0..19 (Class 0), right partition 20..39 (Class 1)
    // Ensures left side has all 0s and right side has all 1s
    const paddingX = 130;
    const paddingY = 40;
    const leftX = paddingX;
    const rightX = CANVAS_WIDTH - paddingX;
    const spacing = (CANVAS_HEIGHT - paddingY * 2) / 19;
    
    // Layout and Nodes
    for (let i = 0; i < 40; i++) {
        const isRight = i >= 20;
        const index = isRight ? i - 20 : i;
        const x = isRight ? rightX : leftX;
        const y = paddingY + index * spacing;
        const label = isRight ? 1 : 0; // Left column nodes = 0, Right column nodes = 1
        
        simulatorState.nodes.push({
            id: i,
            x: x,
            y: y,
            label: label,
            isQueried: false,
            predProb: 0.5
        });
    }
    
    // Build edges from precomputed data (strictly bipartite, no intra-column edges)
    PRECOMPUTED_BIPARTITE_EDGES.forEach(e => {
        simulatorState.edges.push({ u: e[0], v: e[1] });
    });
}

function seedInitialLabels() {
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

// ---------------------------------------------------------
// 2. Label Propagation Solvers
// ---------------------------------------------------------
function runPropagation() {
    const propType = simulatorState.propagatorType;
    
    if (propType === 'harmonic') {
        solveHarmonicLP(simulatorState.labeledSet);
    } else if (propType === 'mincut') {
        solveMinCutLP(simulatorState.labeledSet);
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
    const rNode = (type === 'bipartite') ? 10 : 6;
    
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
    const clickLimit = (type === 'bipartite') ? 20 : 12;
    
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
    const hoverLimit = (type === 'bipartite') ? 15 : 10;
    
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
        const maxRisk = (simulatorState.graphType === 'bipartite') ? 20 : 200;
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
