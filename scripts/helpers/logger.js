class Logger {
    static log(message, data = null) {
        if (data) {
            console.log(message, data);
        } else {
            console.log(message);
        }
    }

    static error(message, error = null) {
        if (error) {
            console.error(message, error);
        } else {
            console.error(message);
        }
    }

    static warn(message, data = null) {
        if (data) {
            console.warn(message, data);
        } else {
            console.warn(message);
        }
    }
}