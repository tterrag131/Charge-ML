// In helpers/ui-helpers.js
class UIHelpers {
    static showLoading() {
        const loader = document.getElementById('loader');
        if (loader) loader.style.display = 'block';
    }

    static hideLoading() {
        const loader = document.getElementById('loader');
        if (loader) loader.style.display = 'none';
    }

    static showError(message) {
        console.error(message);
        // Add your error display logic here
    }

    static highlightElement(element, duration = 1000) {
        if (element) {
            element.classList.add('updated');
            setTimeout(() => {
                element.classList.remove('updated');
            }, duration);
        }
    }

    static updateElement(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = value;
            this.highlightElement(element);
        }
    }
}
