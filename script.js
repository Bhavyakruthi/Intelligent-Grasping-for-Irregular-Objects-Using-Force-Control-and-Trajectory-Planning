// Function to show a specific section and hide others
function showSection(sectionId) {
    const sections = document.querySelectorAll("main > section");
    sections.forEach(section => {
        section.style.display = (section.id === sectionId) ? "block" : "none";
    });
}

// Show 'abstract' section by default on page load
window.onload = () => {
    const defaultSection = "abstract";
    showSection(defaultSection);
};

function launchSimulation() {
    alert('Launching Simulation...');
}

// Scroll animation for timeline items
window.addEventListener('scroll', function () {
    const timelineItems = document.querySelectorAll('.timeline-item');
    timelineItems.forEach(item => {
        const rect = item.getBoundingClientRect();
        if (rect.top >= 0 && rect.top <= window.innerHeight) {
            item.classList.add('visible');
        }
    });
});
