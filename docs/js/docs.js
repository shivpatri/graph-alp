/**
 * GraphALP Documentation JavaScript
 * Implements sidebar search filtering, code block copy-to-clipboard,
 * smooth scrolling highlight tracking, and GIF animation restarts.
 */

// 1. Copy Code Snippets to Clipboard
function copyCode(button) {
    const codeBlock = button.closest('.code-block-wrapper').querySelector('pre');
    const textToCopy = codeBlock.innerText;

    navigator.clipboard.writeText(textToCopy).then(() => {
        // Visual feedback
        const originalHTML = button.innerHTML;
        button.innerHTML = '✓ Copied!';
        button.style.color = '#34d399'; // green

        setTimeout(() => {
            button.innerHTML = originalHTML;
            button.style.color = '';
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy text: ', err);
    });
}

// 2. Real-time Search Filter for Sidebar and Main Content Sections
function filterDocs() {
    const query = document.getElementById('docs-search').value.toLowerCase();
    const navLinks = document.querySelectorAll('.docs-sidebar .nav-section ul li');
    const sections = document.querySelectorAll('.docs-content .docs-section');
    const classBlocks = document.querySelectorAll('.docs-content .api-class-block');

    // Filter sidebar list items
    navLinks.forEach(item => {
        const text = item.textContent.toLowerCase();
        if (text.includes(query)) {
            item.style.display = 'block';
        } else {
            item.style.display = 'none';
        }
    });

    // Filter main API blocks and class signatures
    classBlocks.forEach(block => {
        const text = block.textContent.toLowerCase();
        if (text.includes(query)) {
            block.style.display = 'block';
        } else {
            block.style.display = 'none';
        }
    });

    // Filter main sections (Getting Started, Label Prop, Active Learning, Benchmarks)
    sections.forEach(section => {
        const title = section.querySelector('h2').textContent.toLowerCase();
        const paragraphs = section.textContent.toLowerCase();
        if (title.includes(query) || paragraphs.includes(query)) {
            section.style.display = 'block';
        } else {
            section.style.display = 'none';
        }
    });
}

// 3. Highlight Active Sidebar Links on Scroll (Intersection Observer)
document.addEventListener('DOMContentLoaded', () => {
    const sections = document.querySelectorAll('.docs-section');
    const navLinks = document.querySelectorAll('.docs-sidebar .nav-section a');

    const observerOptions = {
        root: null,
        rootMargin: '0px -10% -80% 0px', // Trigger when section is in upper-middle of viewport
        threshold: 0
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const id = entry.target.getAttribute('id');
                
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${id}`) {
                        link.classList.add('active');
                    }
                });

                // Restart any educational GIFs in this section for the user
                restartGifs(entry.target);
            }
        });
    }, observerOptions);

    sections.forEach(section => {
        observer.observe(section);
    });
});

// Helper to restart GIFs when they are scrolled into view or clicked
function restartGifs(container) {
    const gifs = container.querySelectorAll('.demo-gif, .benchmark-chart');
    gifs.forEach(gif => {
        // If not already restarted recently
        if (!gif.dataset.restarted || (Date.now() - parseInt(gif.dataset.restarted)) > 5000) {
            const src = gif.getAttribute('src');
            gif.setAttribute('src', '');
            setTimeout(() => {
                const cleanSrc = src.split('?')[0];
                gif.setAttribute('src', `${cleanSrc}?t=${Date.now()}`);
                gif.dataset.restarted = Date.now();
            }, 50);
        }
    });
}
