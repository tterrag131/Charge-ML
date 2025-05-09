class PDPCalculator {
    constructor() {
        // Configuration defaults
        this.config = {
            rates: {
                pick: {
                    default: 300,
                    current: 300
                },
                transship: {
                    default: 8000,
                    current: 8000
                },
                singles: {
                    sioc: { default: 350, current: 350 },
                    smallPack: { default: 180, current: 180 },
                    mixPack: { default: 70, current: 70 },
                    smartPac: { default: 450, current: 450 },
                    smartPacPoly: { default: 450, current: 450 }
                },
                afe: {
                    induct: { default: 1160, current: 1160 },
                    rebin: { default: 560, current: 560 },
                    pack: { default: 200, current: 200 }
                }
            },
            splits: {
                multi: { default: 0.65, current: 0.65 },
                afe1: { default: 0.55, current: 0.55 },
                smartpac: { default: 0.40, current: 0.40 }
            },
            shiftHours: {
                dsFirst: 6,
                dsSecond: 5.5,
                nsFirst: 5,
                nsSecond: 6
            },
            productivityFactor: 0.75
        };

        // Volume data storage
        this.volumeData = null;
        this.multiPercentage = 0;

        // Initialize when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.initialize());
        } else {
            this.initialize();
        }
    }

    async initialize() {
        try {
            setupThemeToggle(); // Add this line
            await this.setupEventListeners();
            this.loadSavedConfiguration();
            this.setupInitialValues();
            await this.startDataPolling();
        } catch (error) {
            Logger.error('Initialization error:', error);
        }
    }
    setupEventListeners() {
        // Calculate button
        const calculateButton = document.getElementById('calculate-pdp');
        if (calculateButton) {
            calculateButton.addEventListener('click', () => this.performCalculations());
        }

        // Reset buttons
        const resetRatesButton = document.getElementById('reset-rates');
        if (resetRatesButton) {
            resetRatesButton.addEventListener('click', () => this.resetRates());
        }

        // Setup rate input listeners
        this.setupRateInputListeners();
        this.setupSplitInputListeners();
    }

    setupRateInputListeners() {
        // Pick and Transship
        this.setupInputListener('pick-rate', 'rates.pick.current');
        this.setupInputListener('transship-rate', 'rates.transship.current');

        // Singles Processing
        const singlesTypes = ['sioc', 'smallPack', 'mixPack', 'smartPac', 'smartPacPoly'];
        singlesTypes.forEach(type => {
            const elementId = `${type.toLowerCase()}-rate`;
            this.setupInputListener(elementId, `rates.singles.${type}.current`);
        });
    }

    setupSplitInputListeners() {
        // Volume splits
        this.setupInputListener('multi-split', 'splits.multi.current', value => value / 100);
        this.setupInputListener('afe1-split', 'splits.afe1.current', value => value / 100);
        this.setupInputListener('smartpac-split', 'splits.smartpac.current', value => value / 100);
    }

    setupInputListener(elementId, configPath, transformer = value => value) {
        const element = document.getElementById(elementId);
        if (element) {
            element.addEventListener('change', (e) => {
                const value = parseFloat(e.target.value);
                if (!isNaN(value) && value >= 0) {
                    this.updateConfig(configPath, transformer(value));
                }
            });
        }
    }

    updateConfig(path, value) {
        const parts = path.split('.');
        let current = this.config;
        for (let i = 0; i < parts.length - 1; i++) {
            current = current[parts[i]];
        }
        current[parts[parts.length - 1]] = value;
        this.saveConfiguration();
    }
    // Continue from previous PDPCalculator class
    async startDataPolling() {
        await this.fetchAndProcessData();
        setInterval(() => this.fetchAndProcessData(), CONSTANTS.REFRESH_INTERVAL);
    }

    async fetchAndProcessData() {
        try {
            UIHelpers.showLoading();
            const data = await this.fetchLatestData();
            if (data) {
                this.volumeData = data;
                this.calculateMultiPercentage();
                this.updateVolumeDisplay();
                UIHelpers.hideLoading();
            }
        } catch (error) {
            Logger.error('Data processing error:', error);
            UIHelpers.hideLoading();
        }
    }

    async fetchLatestData(maxAttempts = 24) {
        const timeInfo = this.getCorrectDateTime();
        let currentAttempt = 0;
        let currentTime = new Date(timeInfo.fullDateTime);

        while (currentAttempt < maxAttempts) {
            try {
                const dateStr = this.formatDateForUrl(currentTime);
                const hour = currentTime.getHours();
                const s3Url = this.constructS3Url(dateStr, hour);
                
                Logger.log('Attempting fetch from:', s3Url);
                
                const response = await fetch(s3Url, {
                    method: 'GET',
                    mode: 'cors',
                    headers: { 'Accept': 'application/json' }
                });
                
                if (response.ok) {
                    const data = await response.json();
                    if (DataValidator.validateData(data)) {
                        return data;
                    }
                }
                
                currentTime.setHours(currentTime.getHours() - 1);
                currentAttempt++;
                
            } catch (error) {
                Logger.error(`Fetch attempt ${currentAttempt + 1} failed:`, error);
                currentTime.setHours(currentTime.getHours() - 1);
                currentAttempt++;
            }
        }
        return null;
    }

    // Volume Calculation Methods
    performCalculations() {
        if (!this.volumeData) {
            UIHelpers.showError('No volume data available');
            return;
        }

        const volumes = this.calculateShiftVolumes();
        this.updateAllHeadcounts(volumes);
        this.updateRollingForecast();  // Add this line
        this.highlightUpdatedValues();
    }

    calculateShiftVolumes() {
        if (!this.volumeData?.next_day?.sarima_predictions) {
            return this.getEmptyVolumes();
        }

        const predictions = this.volumeData.next_day.sarima_predictions;
        
        return {
            dsFirst: this.calculateShiftTotal(predictions, 6, 12),
            dsSecond: this.calculateShiftTotal(predictions, 12, 18),
            nsFirst: this.calculateShiftTotal(predictions, 18, 23),
            nsSecond: this.calculateShiftTotal(predictions, 0, 6)
        };
    }

    calculateShiftTotal(predictions, startHour, endHour) {
        let totalVolume = 0;
        
        for (let hour = startHour; hour < endHour; hour++) {
            if (predictions[hour] && predictions[hour - 1]) {
                const currentVolume = predictions[hour].Predicted_Workable;
                const previousVolume = predictions[hour - 1].Predicted_Workable;
                const newVolume = Math.max(0, currentVolume - previousVolume);
                totalVolume += newVolume;
            } else if (predictions[hour]) {
                totalVolume += predictions[hour].Predicted_Workable;
            }
        }
        
        return totalVolume;
    }

    calculateMultiPercentage() {
        if (this.volumeData?.Ledger_Information?.metrics) {
            const { IPTM, IPTNW } = this.volumeData.Ledger_Information.metrics;
            if (IPTM[0] && IPTNW[0]) {
                this.multiPercentage = IPTM[0] / IPTNW[0];
            }
        }
    }

    // Display Update Methods
    updateVolumeDisplay() {
        const volumes = this.calculateShiftVolumes();
        const totalVolume = Object.values(volumes).reduce((a, b) => a + b, 0);
        
        // Update total volume
        const totalElement = document.getElementById('total-volume-display');
        if (totalElement) {
            totalElement.textContent = totalVolume.toLocaleString();
        }
    
        // Calculate volumes
        const multiVolume = Math.round(totalVolume * this.multiPercentage);
        const singlesVolume = totalVolume - multiVolume;
    
        // Update volume displays
        const elements = {
            multi: document.getElementById('multi-volume'),
            singles: document.getElementById('singles-volume'),
            multiPercent: document.getElementById('multi-percentage'),
            singlesPercent: document.getElementById('singles-percentage')
        };
    
        // Update volumes
        if (elements.multi) {
            elements.multi.textContent = multiVolume.toLocaleString();
        }
        if (elements.singles) {
            elements.singles.textContent = singlesVolume.toLocaleString();
        }
    
        // Update percentages
        if (elements.multiPercent) {
            elements.multiPercent.textContent = 
                `${(this.multiPercentage * 100).toFixed(1)}%`;
        }
        if (elements.singlesPercent) {
            elements.singlesPercent.textContent = 
                `${((1 - this.multiPercentage) * 100).toFixed(1)}%`;
        }
    
        // Log volume information
        Logger.log('Volume Update:', {
            total: totalVolume,
            multi: multiVolume,
            singles: singlesVolume,
            multiPercentage: this.multiPercentage
        });
    }
    

    updateRollingForecast() {
        if (!this.volumeData?.extended_predictions?.predictions) {
            Logger.warn('No extended predictions available for rolling forecast');
            return;
        }
    
        const predictions = this.volumeData.extended_predictions.predictions;
        const currentTime = new Date();
        const currentHour = currentTime.getHours();
        
        // Determine current shift
        const isDayShift = currentHour >= 6 && currentHour < 18;
        const currentShiftStart = isDayShift ? 6 : 18;
        
        // Get next 4 shifts worth of data
        const shifts = [];
        let startTime = new Date(currentTime);
        startTime.setHours(currentShiftStart, 0, 0, 0);
    
        if (currentTime < startTime) {
            startTime.setDate(startTime.getDate() - 1);
        }
    
        // Calculate volumes for next 4 shifts
        for (let i = 0; i < 4; i++) {
            const shiftStartTime = new Date(startTime);
            const shiftEndTime = new Date(startTime);
            shiftEndTime.setHours(startTime.getHours() + 12);
    
            const isDay = shiftStartTime.getHours() === 6;
            const shiftVolume = this.calculateShiftVolume(
                predictions,
                shiftStartTime,
                shiftEndTime
            );
    
            shifts.push({
                startTime: shiftStartTime,
                endTime: shiftEndTime,
                volume: shiftVolume,
                label: isDay ? 'Day Shift' : 'Night Shift'
            });
    
            startTime.setHours(startTime.getHours() + 12);
        }
    
        // Update the UI
        this.updateRollingForecastDisplay(shifts);
    }
    
    calculateShiftVolume(predictions, startTime, endTime) {
        const isNightShift = startTime.getHours() === 18;
        let volume = 0;
    
        if (isNightShift) {
            // Night shift (18:00 - 06:00 next day)
            // First part (18:00 - 23:00)
            const endOfDayVolume = this.findVolumeAtHour(predictions, startTime, 23);
            const startVolume = this.findVolumeAtHour(predictions, startTime, 18);
            const firstPart = endOfDayVolume - startVolume;
    
            // Second part (00:00 - 06:00 next day)
            const nextDayDate = new Date(startTime);
            nextDayDate.setDate(nextDayDate.getDate() + 1);
            const endVolume = this.findVolumeAtHour(predictions, nextDayDate, 6);
            const secondPart = endVolume; // Since it starts from 0 at midnight
    
            volume = firstPart + secondPart;
        } else {
            // Day shift (06:00 - 18:00)
            const endVolume = this.findVolumeAtHour(predictions, startTime, 18);
            const startVolume = this.findVolumeAtHour(predictions, startTime, 6);
            volume = endVolume - startVolume;
        }
    
        return volume;
    }
    
    findVolumeAtHour(predictions, date, hour) {
        const searchTime = `${date.getFullYear()}-${(date.getMonth() + 1).toString().padStart(2, '0')}-${date.getDate().toString().padStart(2, '0')}T${hour.toString().padStart(2, '0')}:00`;
        
        const prediction = predictions.find(p => p.Time === searchTime);
        return prediction ? prediction.Predicted_Workable : 0;
    }
    
    updateRollingForecastDisplay(shifts) {
        // Find max volume for scaling
        const maxVolume = Math.max(...shifts.map(s => s.volume));
    
        shifts.forEach((shift, index) => {
            const barElement = document.getElementById(index === 0 ? 'current-shift' : `next-shift-${index}`);
            if (!barElement) return;
    
            const fillElement = barElement.querySelector('.bar-fill');
            const labelElement = barElement.querySelector('.bar-label');
    
            if (fillElement && labelElement) {
                const percentage = (shift.volume / maxVolume * 100).toFixed(1);
                fillElement.style.height = `${percentage}%`;
                
                labelElement.innerHTML = `
                    ${shift.label}<br>
                    ${this.formatNumber(shift.volume)} units<br>
                    ${shift.startTime.getHours().toString().padStart(2, '0')}:00 - 
                    ${shift.endTime.getHours().toString().padStart(2, '0')}:00
                `;
            }
    
            // Add shift timing tooltip
            barElement.title = `${shift.label}
    Volume: ${this.formatNumber(shift.volume)} units
    ${shift.startTime.getHours().toString().padStart(2, '0')}:00 - ${shift.endTime.getHours().toString().padStart(2, '0')}:00`;
        });
    }
    
    

    updateVolumeElement(element, volume) {
        if (element) {
            element.textContent = volume.toLocaleString();
        }
    }

    updatePercentageElements(multiElement, singlesElement) {
        if (multiElement) {
            multiElement.textContent = `${(this.multiPercentage * 100).toFixed(1)}%`;
        }
        if (singlesElement) {
            singlesElement.textContent = `${((1 - this.multiPercentage) * 100).toFixed(1)}%`;
        }
    }

    // Utility Methods
    formatDateForUrl(date) {
        const year = date.getFullYear();
        const month = (date.getMonth() + 1).toString().padStart(2, '0');
        const day = date.getDate().toString().padStart(2, '0');
        return `${year}-${month}-${day}`;
    }

    constructS3Url(dateStr, hour) {
        return `https://${CONSTANTS.BUCKET_NAME}.s3.${CONSTANTS.AWS_REGION}.amazonaws.com/predictions/${dateStr}_${hour.toString().padStart(2, '0')}/VIZ.json`;
    }

    getEmptyVolumes() {
        return {
            dsFirst: 0,
            dsSecond: 0,
            nsFirst: 0,
            nsSecond: 0
        };
    }

    getCorrectDateTime() {
        return {
            fullDateTime: new Date().toISOString()
        };
    }

        // Continue from previous PDPCalculator class
    // Headcount Calculation Methods
    updateAllHeadcounts(volumes) {
        const shifts = ['ds1', 'ds2', 'ns1', 'ns2'];
        const shiftVolumes = {
            ds1: { volume: volumes.dsFirst, hours: this.config.shiftHours.dsFirst },
            ds2: { volume: volumes.dsSecond, hours: this.config.shiftHours.dsSecond },
            ns1: { volume: volumes.nsFirst, hours: this.config.shiftHours.nsFirst },
            ns2: { volume: volumes.nsSecond, hours: this.config.shiftHours.nsSecond }
        };

        shifts.forEach(shift => {
            this.calculateAndUpdatePickHeadcount(shift, shiftVolumes[shift]);
            this.calculateAndUpdateAFEHeadcount('afe1', shift, shiftVolumes[shift]);
            this.calculateAndUpdateAFEHeadcount('afe2', shift, shiftVolumes[shift]);
            this.calculateAndUpdateSinglesHeadcount(shift, shiftVolumes[shift]);
        });
    }

    // Pick Department Calculations
    calculateAndUpdatePickHeadcount(shiftKey, shiftData) {
        const { volume, hours } = shiftData;
        
        // Calculate tranship volume for this shift period
        const transhipVolume = this.config.rates.transship.current * hours;
        
        // Adjust total volume for productivity
        const adjustedVolume = (volume + transhipVolume) / this.config.productivityFactor;
        
        // Calculate units per hour and headcount
        const unitsPerHour = adjustedVolume / hours;
        const headcount = Math.ceil(unitsPerHour / this.config.rates.pick.current);

        // Update display
        const element = document.getElementById(`pick-${shiftKey}-hc`);
        if (element) {
            element.textContent = headcount;
            this.markElementForHighlight(element);
        }

        // Update tranship contribution display if element exists
        const transhipElement = document.getElementById(`pick-${shiftKey}-tranship`);
        if (transhipElement) {
            transhipElement.textContent = 
                `+${Math.round(transhipVolume).toLocaleString()} tranship`;
        }

        return headcount;
    }

    // AFE Calculations
    calculateAndUpdateAFEHeadcount(afeType, shiftKey, shiftData) {
        const { volume, hours } = shiftData;
        const isAfe1 = afeType === 'afe1';
        const afeSplit = isAfe1 ? this.config.splits.afe1.current : (1 - this.config.splits.afe1.current);
        
        // Calculate volume for this AFE
        const multiVolume = volume * this.config.splits.multi.current;
        const afeVolume = multiVolume * afeSplit;
        
        // Adjust for productivity
        const adjustedVolume = afeVolume / this.config.productivityFactor;
        const unitsPerHour = adjustedVolume / hours;

        // Calculate headcount for each station
        const headcounts = {
            induct: Math.ceil(unitsPerHour / this.config.rates.afe.induct.current),
            rebin: Math.ceil(unitsPerHour / this.config.rates.afe.rebin.current),
            pack: Math.ceil(unitsPerHour / this.config.rates.afe.pack.current)
        };

        // Update display for each station
        ['induct', 'rebin', 'pack'].forEach(station => {
            const element = document.getElementById(`${afeType}-${shiftKey}-${station}`);
            if (element) {
                element.textContent = headcounts[station];
                this.markElementForHighlight(element);
            }
        });

        // Calculate and update total
        const totalHC = Object.values(headcounts).reduce((sum, count) => sum + count, 0);
        const totalElement = document.getElementById(`${afeType}-${shiftKey}-total`);
        if (totalElement) {
            totalElement.textContent = totalHC;
            this.markElementForHighlight(totalElement);
        }

        return headcounts;
    }

    // Singles Department Calculations
    calculateAndUpdateSinglesHeadcount(shiftKey, shiftData) {
        const { volume, hours } = shiftData;
        const singlesVolume = volume * (1 - this.config.splits.multi.current);
        const smartPacVolume = singlesVolume * this.config.splits.smartpac.current;
        const regularSinglesVolume = singlesVolume - smartPacVolume;

        // Adjust volumes for productivity
        const adjustedSmartPacVolume = smartPacVolume / this.config.productivityFactor;
        const adjustedRegularVolume = regularSinglesVolume / this.config.productivityFactor;

        // Calculate units per hour
        const smartPacUnitsPerHour = adjustedSmartPacVolume / hours;
        const regularUnitsPerHour = adjustedRegularVolume / hours;

        // Calculate headcount for each type
        const headcounts = {
            sioc: Math.ceil(regularUnitsPerHour * 0.3 / this.config.rates.singles.sioc.current),
            smallPack: Math.ceil(regularUnitsPerHour * 0.4 / this.config.rates.singles.smallPack.current),
            mixPack: Math.ceil(regularUnitsPerHour * 0.3 / this.config.rates.singles.mixPack.current),
            smartPac: Math.ceil(smartPacUnitsPerHour * 0.5 / this.config.rates.singles.smartPac.current),
            smartPacPoly: Math.ceil(smartPacUnitsPerHour * 0.5 / this.config.rates.singles.smartPacPoly.current)
        };

        // Update display for each type
        Object.entries(headcounts).forEach(([type, hc]) => {
            const element = document.getElementById(`${type}-${shiftKey}-hc`);
            if (element) {
                element.textContent = hc;
                this.markElementForHighlight(element);
            }
        });

        // Calculate and update total
        const totalHC = Object.values(headcounts).reduce((sum, count) => sum + count, 0);
        const totalElement = document.getElementById(`singles-${shiftKey}-total`);
        if (totalElement) {
            totalElement.textContent = totalHC;
            this.markElementForHighlight(totalElement);
        }

        return headcounts;
    }

    // Highlighting Methods
    markElementForHighlight(element) {
        element.classList.add('pending-update');
    }

    highlightUpdatedValues() {
        const elements = document.querySelectorAll('.pending-update');
        elements.forEach(element => {
            element.classList.remove('pending-update');
            element.classList.add('updated');
            setTimeout(() => {
                element.classList.remove('updated');
            }, 1000);
        });
    }

    // Continue from previous PDPCalculator class
    // Configuration Management Methods
    loadSavedConfiguration() {
        try {
            const savedConfig = localStorage.getItem('pdpConfig');
            if (savedConfig) {
                const parsedConfig = JSON.parse(savedConfig);
                this.mergeConfiguration(parsedConfig);
            }
        } catch (error) {
            Logger.error('Error loading saved configuration:', error);
            this.resetToDefaults();
        }
    }

    mergeConfiguration(savedConfig) {
        // Merge rates
        Object.entries(savedConfig.rates || {}).forEach(([category, rates]) => {
            if (this.config.rates[category]) {
                Object.entries(rates).forEach(([rate, value]) => {
                    if (this.config.rates[category][rate]) {
                        this.config.rates[category][rate].current = value.current;
                    }
                });
            }
        });

        // Merge splits
        Object.entries(savedConfig.splits || {}).forEach(([split, value]) => {
            if (this.config.splits[split]) {
                this.config.splits[split].current = value.current;
            }
        });
    }

    saveConfiguration() {
        try {
            localStorage.setItem('pdpConfig', JSON.stringify(this.config));
        } catch (error) {
            Logger.error('Error saving configuration:', error);
        }
    }

    setupInitialValues() {
        this.setupRateInputValues();
        this.setupSplitInputValues();
        this.setupTransshipValue();
    }

    setupRateInputValues() {
        // Pick and AFE rates
        this.setInputValue('pick-rate', this.config.rates.pick.current);
        
        // Singles rates
        Object.entries(this.config.rates.singles).forEach(([type, config]) => {
            this.setInputValue(`${type.toLowerCase()}-rate`, config.current);
        });

        // AFE station rates
        Object.entries(this.config.rates.afe).forEach(([station, config]) => {
            this.setInputValue(`afe-${station}-rate`, config.current);
        });
    }

    setupSplitInputValues() {
        this.setInputValue('multi-split', this.config.splits.multi.current * 100);
        this.setInputValue('afe1-split', this.config.splits.afe1.current * 100);
        this.setInputValue('smartpac-split', this.config.splits.smartpac.current * 100);
        
        // Update AFE2 display
        const afe2Element = document.getElementById('afe2-split');
        if (afe2Element) {
            afe2Element.textContent = (100 - (this.config.splits.afe1.current * 100)).toFixed(0);
        }
    }

    setupTransshipValue() {
        this.setInputValue('transship-rate', this.config.rates.transship.current);
    }

    // Reset Methods
    resetToDefaults() {
        this.resetRates();
        this.resetSplits();
        this.saveConfiguration();
        this.setupInitialValues();
        this.performCalculations();
    }

    resetRates() {
        // Reset pick and transship rates
        this.config.rates.pick.current = this.config.rates.pick.default;
        this.config.rates.transship.current = this.config.rates.transship.default;

        // Reset singles rates
        Object.keys(this.config.rates.singles).forEach(type => {
            this.config.rates.singles[type].current = 
                this.config.rates.singles[type].default;
        });

        // Reset AFE rates
        Object.keys(this.config.rates.afe).forEach(station => {
            this.config.rates.afe[station].current = 
                this.config.rates.afe[station].default;
        });
    }

    resetSplits() {
        Object.keys(this.config.splits).forEach(split => {
            this.config.splits[split].current = this.config.splits[split].default;
        });
    }

    // Utility Methods
    setInputValue(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            element.value = value;
        }
    }

    validateNumericInput(value, min = 0, max = Infinity) {
        const numValue = parseFloat(value);
        if (isNaN(numValue)) return false;
        return numValue >= min && numValue <= max;
    }

    formatNumber(number, decimals = 0) {
        return number.toLocaleString(undefined, {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        });
    }

    // Error Handling Methods
    handleError(error, context) {
        Logger.error(`Error in ${context}:`, error);
        UIHelpers.showError(`An error occurred in ${context}. Please try again.`);
    }
}
// Move this function outside of the PDPCalculator class
function setupThemeToggle() {
    const themeToggle = document.getElementById('themeToggle');
    const themeIcon = themeToggle.querySelector('.theme-toggle-icon');
    
    // Function to set theme
    function setTheme(isDark) {
        document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
        themeIcon.textContent = isDark ? '☀️' : '🌙';
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
    }

    // Check for saved theme preference or default to user's system preference
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (savedTheme) {
        setTheme(savedTheme === 'dark');
    } else {
        setTheme(prefersDark);
    }

    // Theme toggle click handler
    themeToggle.addEventListener('click', () => {
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        setTheme(!isDark);
    });

    // Optional: Listen for system theme changes
    window.matchMedia('(prefers-color-scheme: dark)').addListener((e) => {
        if (!localStorage.getItem('theme')) {
            setTheme(e.matches);
        }
    });
}


// Initialize the calculator
const pdpCalculator = new PDPCalculator();
