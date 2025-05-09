// theme.js
document.addEventListener('DOMContentLoaded', () => {
    const themeToggle = document.getElementById('themeToggle');
    const themeIcon = themeToggle.querySelector('.theme-toggle-icon');
    
    function setTheme(isDark) {
        // Apply to HTML element
        document.documentElement.classList.toggle('dark-theme', isDark);
        
        // Update icon
        themeIcon.textContent = isDark ? '☀️' : '🌙';
        
        // Save preference
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
        
        // Debug logging (you can remove these later)
        console.log('Theme changed:', isDark ? 'dark' : 'light');
        console.log('HTML classes:', document.documentElement.className);
        console.log('Body background:', getComputedStyle(document.body).backgroundColor);
        console.log('HTML background:', getComputedStyle(document.documentElement).backgroundColor);
    }

    // Check system preference
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)');
    
    // Initialize theme based on saved preference or system preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        setTheme(savedTheme === 'dark');
    } else {
        setTheme(prefersDark.matches);
    }

    // Toggle theme on button click
    themeToggle.addEventListener('click', () => {
        const isDark = !document.documentElement.classList.contains('dark-theme');
        setTheme(isDark);
    });

    // Listen for system theme changes
    prefersDark.addEventListener('change', (e) => {
        if (!localStorage.getItem('theme')) {
            setTheme(e.matches);
        }
    });
});
