/**
 * GraphALP Documentation JavaScript
 * Logic for responsive tabs, dynamic UI interactions, and visualization restarts.
 */

function switchTab(tabName) {
    // 1. Get all tabs and contents
    const tabs = document.querySelectorAll('.tab-btn');
    const contents = document.querySelectorAll('.tab-content');
    
    // 2. Remove active classes from all tabs and contents
    tabs.forEach(tab => {
        tab.classList.remove('active');
    });
    
    contents.forEach(content => {
        content.classList.remove('active-content');
    });
    
    // 3. Find the target tab and add active state
    const activeBtn = document.getElementById(`tab-btn-${tabName}`);
    if (activeBtn) {
        activeBtn.classList.add('active');
    }
    
    // 4. Find target content panel and display it
    const activeContent = document.getElementById(`section-${tabName}`);
    if (activeContent) {
        activeContent.classList.add('active-content');
        
        // 5. Restart GIFs so the user watches the animation from step 0 when selecting the tab!
        restartGifs(activeContent);
        
        // 6. Redraw simulator graph if activating the simulator tab
        if (tabName === 'simulator' && typeof drawGraph === 'function') {
            drawGraph();
        }
    }
}

/**
 * Force-reloads all GIFs inside a container so their animations restart from frame 0.
 * This is a highly requested polish for educational GIFs, ensuring the user actually sees 
 * the beginning of the sequence when they click into a tab.
 */
function restartGifs(container) {
    const gifs = container.querySelectorAll('.demo-gif, .benchmark-chart');
    gifs.forEach(gif => {
        const src = gif.getAttribute('src');
        // Clear the source and append a tiny query parameter timestamp to force a browser refresh of the GIF
        gif.setAttribute('src', '');
        setTimeout(() => {
            const cleanSrc = src.split('?')[0];
            gif.setAttribute('src', `${cleanSrc}?t=${Date.now()}`);
        }, 10);
    });
}

// Initial trigger to restart any visible GIFs on page load
document.addEventListener('DOMContentLoaded', () => {
    const activeContent = document.querySelector('.tab-content.active-content');
    if (activeContent) {
        restartGifs(activeContent);
    }
});
