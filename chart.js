// constants.js
const CONSTANTS = {
    AWS_REGION: 'us-west-1',
    IDENTITY_POOL_ID: 'us-west-1:06c9cd9c-b1c0-4252-8c0c-379171b51348',
    BUCKET_NAME: 'ledger-prediction-charting-008971633421',
    REFRESH_INTERVAL: 3600000,
    CHART_COLORS: {
        primary: '#3498db', 
        secondary: '#2ecc71',  
        accent: '#e74c3c',      
        neutral: '#95a5a6',    
        purple: '#9b59b6',
        yellow: '#f1c40f'
    }
};

// Chart Configuration
const ChartConfig = {
    defaultOptions: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: {
                beginAtZero: true,
                title: {
                    display: true,
                    text: 'Workable Count'
                }
            },
            x: {
                title: {
                    display: true,
                    text: 'Time'
                }
            }
        },
        plugins: {
            legend: {
                position: 'bottom',
                labels: {
                    padding: 20,
                    usePointStyle: true,
                    pointStyle: 'circle'
                }
            },
            tooltip: {
                mode: 'index',
                intersect: false,
                callbacks: {
                    label: function(context) {
                        const value = context.raw;
                        return `${context.dataset.label}: ${value ? value.toLocaleString() : 'No data'}`;
                    }
                }
            }
        },
        interaction: {
            mode: 'nearest',
            axis: 'x',
            intersect: false
        }
    }
};

// Global state
let charts = {};
let globalData = null;
let isShowingTomorrow = false;
let currentMetricsData = null;
let nextDayMetricsData = null;
let nextDaySarimaFinal = 0;

// Logger utility
const Logger = {
    log: (message, data) => console.log(message, data),
    error: (message, error) => console.error(message, error),
    warn: (message) => console.warn(message)
};

// Error Handler
function safeExecute(fn, errorMessage) {
    try {
        return fn();
    } catch (error) {
        Logger.error(errorMessage, error);
        UIHelpers.displayErrorMessage(errorMessage);
        return null;
    }
}

// UI Helpers
const UIHelpers = {
    showLoading() {
        const loadingDiv = document.getElementById('loading-message') || this.createLoadingDiv();
        loadingDiv.style.display = 'block';
    },

    hideLoading() {
        const loadingDiv = document.getElementById('loading-message');
        if (loadingDiv) {
            loadingDiv.style.display = 'none';
        }
    },

    createLoadingDiv() {
        const loadingDiv = document.createElement('div');
        loadingDiv.id = 'loading-message';
        loadingDiv.innerHTML = 'Loading data...';
        loadingDiv.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background-color: rgba(0,0,0,0.8);
            color: white;
            padding: 20px;
            border-radius: 5px;
            z-index: 1000;
        `;
        document.body.appendChild(loadingDiv);
        return loadingDiv;
    },

    displayErrorMessage(message) {
        const errorDiv = document.getElementById('error-message') || this.createErrorDiv();
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
        setTimeout(() => {
            errorDiv.style.display = 'none';
        }, 5000);
    },

    createErrorDiv() {
        const errorDiv = document.createElement('div');
        errorDiv.id = 'error-message';
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background-color: #ff5555;
            color: white;
            padding: 15px;
            border-radius: 5px;
            display: none;
            z-index: 1000;
            animation: fadeIn 0.3s ease-in;
        `;
        document.body.appendChild(errorDiv);
        return errorDiv;
    },

    setupMetricsToggle() {
        const timeToggleBtn = document.getElementById('timeToggleBtn');
        const todayMetrics = document.getElementById('todayMetrics');
        const tomorrowMetrics = document.getElementById('tomorrowMetrics');

        if (!timeToggleBtn || !todayMetrics || !tomorrowMetrics) {
            Logger.error('Required elements for metrics toggle not found');
            return;
        }

        timeToggleBtn.addEventListener('click', () => {
            isShowingTomorrow = !isShowingTomorrow;
            
            timeToggleBtn.querySelector('.toggle-text').textContent = 
                isShowingTomorrow ? 'View Today' : 'View Tomorrow';

            if (isShowingTomorrow) {
                todayMetrics.classList.remove('active');
                tomorrowMetrics.classList.add('active');
                UpdateFunctions.updateTomorrowMetrics();
            } else {
                tomorrowMetrics.classList.remove('active');
                todayMetrics.classList.add('active');
                UpdateFunctions.updateTodayMetrics();
            }
        });
    },

    createRefreshButton() {
        const refreshButton = document.createElement('button');
        refreshButton.textContent = 'Refresh Page';
        refreshButton.onclick = () => location.reload();
        refreshButton.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 10px 20px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        `;
        
        refreshButton.addEventListener('mouseover', () => {
            refreshButton.style.backgroundColor = '#2980b9';
        });
        
        refreshButton.addEventListener('mouseout', () => {
            refreshButton.style.backgroundColor = '#3498db';
        });

        document.body.appendChild(refreshButton);
    }
};
// Data Validator
const DataValidator = {
    validateData(data) {
        const requiredFields = ['current_day', 'next_day', 'Ledger_Information'];
        return requiredFields.every(field => data && data[field]);
    },

    validateMetrics(metrics) {
        return metrics && typeof metrics === 'object' && 
               Object.keys(metrics).length > 0;
    }
};

// Data Processing Module
const DataProcessor = {
    getCorrectDateTime() {
        const now = new Date();
        
        const year = now.getFullYear();
        const month = (now.getMonth() + 1).toString().padStart(2, '0');
        const day = now.getDate().toString().padStart(2, '0');
        const hour = now.getHours();

        const formattedDate = `${year}-${month}-${day}`;

        Logger.log('Time Debug:', {
            localDateTime: now.toLocaleString(),
            formattedDate,
            hour,
            components: { year, month, day, hour }
        });

        return {
            date: formattedDate,
            hour,
            fullDateTime: now
        };
    },

    async fetchData(maxAttempts = 24) {
        UIHelpers.showLoading();
        const timeInfo = this.getCorrectDateTime();
        let currentAttempt = 0;
        let currentTime = new Date(timeInfo.fullDateTime);

        while (currentAttempt < maxAttempts) {
            try {
                const year = currentTime.getFullYear();
                const month = (currentTime.getMonth() + 1).toString().padStart(2, '0');
                const day = currentTime.getDate().toString().padStart(2, '0');
                const hour = currentTime.getHours();
                
                const dateStr = `${year}-${month}-${day}`;
                const s3Url = `https://${CONSTANTS.BUCKET_NAME}.s3.${CONSTANTS.AWS_REGION}.amazonaws.com/predictions/${dateStr}_${hour.toString().padStart(2, '0')}/VIZ.json`;
                
                Logger.log('Attempting fetch from:', s3Url);
                
                const response = await fetch(s3Url, {
                    method: 'GET',
                    mode: 'cors',
                    headers: { 'Accept': 'application/json' }
                });
                
                if (response.ok) {
                    const data = await response.json();
                    Logger.log('Successfully found data at:', s3Url);
                    
                    if (DataValidator.validateData(data)) {
                        UpdateFunctions.updateDashboard(data);
                        UIHelpers.hideLoading();
                        return;
                    }
                }
                
                currentTime.setHours(currentTime.getHours() - 1);
                currentAttempt++;
                
            } catch (error) {
                Logger.error(`Attempt ${currentAttempt + 1} failed:`, error);
                currentTime.setHours(currentTime.getHours() - 1);
                currentAttempt++;
            }
        }

        UIHelpers.hideLoading();
        UIHelpers.displayErrorMessage('Unable to load prediction data');
    },

    transformRFPredictions(rfData, currentDayData) {
        if (!rfData || !currentDayData || currentDayData.length === 0) {
            return rfData;
        }

        try {
            const lastKnownPoint = currentDayData[currentDayData.length - 1];
            if (!lastKnownPoint) return rfData;

            const correspondingRF = rfData.find(rf => rf.Time === lastKnownPoint.Time);
            if (!correspondingRF) return rfData;

            const difference = lastKnownPoint.Workable - correspondingRF.RF_Prediction;

            return rfData.map(point => {
                const pointTime = new Date(point.Time).getTime();
                const lastKnownTime = new Date(lastKnownPoint.Time).getTime();

                if (pointTime <= lastKnownTime) {
                    const actualPoint = currentDayData.find(d => d.Time === point.Time);
                    return {
                        Time: point.Time,
                        RF_Prediction: actualPoint ? actualPoint.Workable : point.RF_Prediction
                    };
                }
                return {
                    Time: point.Time,
                    RF_Prediction: Math.round(point.RF_Prediction + difference)
                };
            });

        } catch (error) {
            Logger.error('Error in transformRFPredictions:', error);
            return rfData;
        }
    }
};

// Metrics Calculator Module
const MetricsCalculator = {
    calculateDayShiftMetrics(data) {
        if (!data || !data.predictions_no_same_day) {
            return {
                morning: 0,
                afternoon: 0,
                total: 0,
                progress: 0
            };
        }

        try {
            const sixAM = data.predictions_no_same_day.find(p => p.Time.includes('T06:00'))?.Predicted_Workable_No_Same_Day || 0;
            const noon = data.predictions_no_same_day.find(p => p.Time.includes('T12:00'))?.Predicted_Workable_No_Same_Day || 0;
            const sixPM = data.predictions_no_same_day.find(p => p.Time.includes('T18:00'))?.Predicted_Workable_No_Same_Day || 0;
            
            const morning = noon - sixAM;
            const afternoon = sixPM - noon;
            const total = sixPM - sixAM;
            
            const currentProgress = data.current_day_data?.length > 0 
                ? data.current_day_data[data.current_day_data.length - 1].Workable - sixAM
                : 0;

            return {
                morning,
                afternoon,
                total,
                progress: currentProgress
            };
        } catch (error) {
            Logger.error('Error calculating day shift metrics:', error);
            return {
                morning: 0,
                afternoon: 0,
                total: 0,
                progress: 0
            };
        }
    },

    calculateNightShiftMetrics(currentDayData, nextDayData) {
        if (!currentDayData?.sarima_predictions || !nextDayData?.sarima_predictions) {
            return {
                tonight: 0,
                tomorrow: 0,
                total: 0
            };
        }

        try {
            const sixPM = currentDayData.sarima_predictions.find(p => p.Time.includes('T18:00'))?.Predicted_Workable || 0;
            const elevenPM = currentDayData.sarima_predictions.find(p => p.Time.includes('T23:00'))?.Predicted_Workable || 0;
            const tonightVolume = elevenPM - sixPM;

            const midnight = nextDayData.sarima_predictions.find(p => p.Time.includes('T00:00'))?.Predicted_Workable || 0;
            const sixAM = nextDayData.sarima_predictions.find(p => p.Time.includes('T06:00'))?.Predicted_Workable || 0;
            const tomorrowVolume = sixAM - midnight;

            return {
                tonight: tonightVolume,
                tomorrow: tomorrowVolume,
                total: tonightVolume + tomorrowVolume
            };
        } catch (error) {
            Logger.error('Error calculating night shift metrics:', error);
            return { tonight: 0, tomorrow: 0, total: 0 };
        }
    },
    calculateTomorrowDayShiftMetrics(data) {
        if (!data || !data.sarima_predictions) {
            return {
                morning: 0,
                afternoon: 0,
                total: 0
            };
        }

        try {
            const sixAM = data.sarima_predictions.find(p => p.Time.includes('T06:00'))?.Predicted_Workable || 0;
            const noon = data.sarima_predictions.find(p => p.Time.includes('T12:00'))?.Predicted_Workable || 0;
            const sixPM = data.sarima_predictions.find(p => p.Time.includes('T18:00'))?.Predicted_Workable || 0;

            return {
                morning: noon - sixAM,
                afternoon: sixPM - noon,
                total: sixPM - sixAM
            };
        } catch (error) {
            Logger.error('Error calculating tomorrow day shift metrics:', error);
            return {
                morning: 0,
                afternoon: 0,
                total: 0
            };
        }
    },

    calculateTomorrowNightShiftMetrics(tomorrowData, currentDayData) {
        if (!tomorrowData?.sarima_predictions) {
            return {
                tomorrow: 0,
                today: 0,
                total: 0
            };
        }

        try {
            // Calculate tomorrow night's portion (18:00-23:00)
            const sixPM = tomorrowData.sarima_predictions.find(p => p.Time.includes('T18:00'))?.Predicted_Workable || 0;
            const elevenPM = tomorrowData.sarima_predictions.find(p => p.Time.includes('T23:00'))?.Predicted_Workable || 0;
            const tomorrowNight = elevenPM - sixPM;

            // Calculate early morning portion (00:00-06:00)
            const midnight = tomorrowData.sarima_predictions.find(p => p.Time.includes('T00:00'))?.Predicted_Workable || 0;
            const sixAM = tomorrowData.sarima_predictions.find(p => p.Time.includes('T06:00'))?.Predicted_Workable || 0;
            const earlyMorning = sixAM - midnight;

            return {
                tomorrow: tomorrowNight,
                today: earlyMorning,
                total: tomorrowNight + earlyMorning
            };
        } catch (error) {
            Logger.error('Error calculating tomorrow night shift metrics:', error);
            return {
                tomorrow: 0,
                today: 0,
                total: 0
            };
        }
    },

    calculateProcessingRate(data) {
        const nw = data.Ledger_Information.metrics.NW;
        const ssf = data.Ledger_Information.metrics.SSF;
        
        if (!nw || !ssf || nw.length < 3) {
            return 0;
        }

        try {
            const differences = [];
            for (let i = 1; i <= 3; i++) {
                const idx = nw.length - i;
                if (idx >= 0) {
                    differences.push(ssf[idx] - nw[idx]);
                }
            }

            return Math.round(differences.reduce((a, b) => a + b, 0) / differences.length);
        } catch (error) {
            Logger.error('Error calculating processing rate:', error);
            return 0;
        }
    }
};

// Add this to your existing code structure
const MultiMetricsCalculator = {
    calculateAverageIPTM(metrics) {
        if (!metrics || !metrics.IPTM) return 0;
        // Filter out zero values and calculate average
        const validIPTM = metrics.IPTM.filter(value => value > 0);
        return validIPTM.length > 0 
            ? validIPTM.reduce((sum, val) => sum + val, 0) / validIPTM.length 
            : 0;
    },

    calculateMultiSplit(value, iptmPercentage) {
        return Math.round(value * (iptmPercentage / 100));
    },

    getMultiTargetDeviation(current, target) {
        if (!target) return 0;
        return ((current - target) / target) * 100;
    },

    calculateMultiMetrics(data) {
        if (!data || !data.Ledger_Information || !data.Ledger_Information.metrics) {
            return {
                avgIPTM: 0,
                currentMultiRate: 0,
                targetMultiRate: 0,
                deviation: 0,
                isOnTarget: false
            };
        }

        const metrics = data.Ledger_Information.metrics;
        const avgIPTM = this.calculateAverageIPTM(metrics);
        
        // Get the latest non-zero MSSF value
        const currentMultiRate = metrics.MSSF
            .filter(val => val > 0)
            .pop() || 0;
        
        return {
            avgIPTM,
            currentMultiRate,
            targetMultiRate: avgIPTM,
            deviation: this.getMultiTargetDeviation(currentMultiRate, avgIPTM),
            isOnTarget: Math.abs(this.getMultiTargetDeviation(currentMultiRate, avgIPTM)) <= 5
        };
    }
};


// Chart Creator Module
const ChartCreator = {
    createPredictionChart(dayData, ledgerData) {
        const ctx = document.getElementById('predictionChart').getContext('2d');
        
        if (charts.predictionChart) {
            charts.predictionChart.destroy();
        }
        
        const transformedRF = DataProcessor.transformRFPredictions(
            dayData.rf_predictions, 
            dayData.current_day_data
        );
        
        const ssfData = dayData.sarima_predictions.map(pred => {
            const hour = parseInt(pred.Time.split('T')[1].split(':')[0]);
            return ledgerData?.metrics?.SSF[hour] || null;
        });
        
        charts.predictionChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: dayData.sarima_predictions.map(d => d.Time),
                datasets: [
                    {
                        label: 'Last Year',
                        data: dayData.previous_year_data.map(d => d.Workable),
                        borderColor: CONSTANTS.CHART_COLORS.neutral,
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4,
                        order: 5
                    },
                    {
                        label: 'SARIMA (No Same Day)',
                        data: dayData.predictions_no_same_day.map(d => d.Predicted_Workable_No_Same_Day),
                        borderColor: CONSTANTS.CHART_COLORS.purple,
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4,
                        order: 4,
                        borderDash: [5, 5]
                    },
                    {
                        label: 'SARIMA Predictions',
                        data: dayData.sarima_predictions.map(d => d.Predicted_Workable),
                        borderColor: CONSTANTS.CHART_COLORS.primary,
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4,
                        order: 3
                    },
                    {
                        label: 'RF Predictions',
                        data: transformedRF.map(d => d.RF_Prediction),
                        borderColor: CONSTANTS.CHART_COLORS.accent,
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4,
                        order: 2
                    },
                    {
                        label: 'Current Day',
                        data: dayData.current_day_data.map(d => d.Workable),
                        borderColor: CONSTANTS.CHART_COLORS.secondary,
                        borderWidth: 3,
                        fill: false,
                        tension: 0.4,
                        order: 1
                    },
                    {
                        label: 'Shipments So Far',
                        data: ssfData,
                        borderColor: CONSTANTS.CHART_COLORS.yellow,
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4,
                        order: 0,
                        pointRadius: 4
                    }
                ]
            },
            options: {
                ...ChartConfig.defaultOptions,
                plugins: {
                    ...ChartConfig.defaultOptions.plugins,
                    title: {
                        display: true,
                        text: `Today's Prediction (${dayData.date})`
                    }
                }
            }
        });
    },

    createTomorrowPredictionChart(nextDayData) {
        if (!nextDayData) return;
        
        const ctx = document.getElementById('tomorrowPredictionChart').getContext('2d');
        nextDaySarimaFinal = nextDayData.sarima_predictions[23].Predicted_Workable;
        
        if (charts.tomorrowPredictionChart) {
            charts.tomorrowPredictionChart.destroy();
        }
        
        charts.tomorrowPredictionChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: nextDayData.sarima_predictions.map(d => d.Time),
                datasets: [
                    {
                        label: 'Last Year',
                        data: nextDayData.previous_year_data.map(d => d.Workable),
                        borderColor: CONSTANTS.CHART_COLORS.neutral,
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4
                    },
                    {
                        label: 'SARIMA Predictions',
                        data: nextDayData.sarima_predictions.map(d => d.Predicted_Workable),
                        borderColor: CONSTANTS.CHART_COLORS.primary,
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4
                    },
                    {
                        label: 'RF Predictions',
                        data: nextDayData.rf_predictions.map(d => d.RF_Prediction),
                        borderColor: CONSTANTS.CHART_COLORS.accent,
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4
                    }
                ]
            },
            options: {
                ...ChartConfig.defaultOptions,
                plugins: {
                    ...ChartConfig.defaultOptions.plugins,
                    title: {
                        display: true,
                        text: `Tomorrow's Prediction (${nextDayData.date})`
                    }
                }
            }
        });
    },

    createBackLogChart(data) {
        const ctx = document.getElementById('backlogChart').getContext('2d');
        
        if (charts.backlogChart) {
            charts.backlogChart.destroy();
        }
        
        const timePoints = data.Ledger_Information.timePoints;
        const eligible = data.Ledger_Information.metrics.Eligible;
        const apu = data.Ledger_Information.metrics.APU;
        const total = eligible.map((e, i) => e + apu[i]);

        charts.backlogChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: timePoints,
                datasets: [
                    {
                        label: 'Eligible',
                        data: eligible,
                        borderColor: CONSTANTS.CHART_COLORS.primary,
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4
                    },
                    {
                        label: 'PBL',
                        data: apu,
                        borderColor: CONSTANTS.CHART_COLORS.secondary,
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4
                    },
                    {
                        label: 'Total',
                        data: total,
                        borderColor: CONSTANTS.CHART_COLORS.yellow,
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4
                    }
                ]
            },
            options: {
                ...ChartConfig.defaultOptions,
                plugins: {
                    ...ChartConfig.defaultOptions.plugins,
                    title: {
                        display: true,
                        text: 'Back Log'
                    }
                }
            }
        });
    },

    createComparisonChart(data) {
        const ctx = document.getElementById('comparisonChart').getContext('2d');
        
        if (charts.comparisonChart) {
            charts.comparisonChart.destroy();
        }
        
        charts.comparisonChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.sarima_predictions.map(d => d.Time),
                datasets: [
                    {
                        label: 'Current Predictions',
                        data: data.sarima_predictions.map(d => d.Predicted_Workable),
                        borderColor: CONSTANTS.CHART_COLORS.primary,
                        fill: false,
                        tension: 0.4
                    },
                    {
                        label: 'Previous Year',
                        data: data.previous_year_data.map(d => d.Workable),
                        borderColor: CONSTANTS.CHART_COLORS.accent,
                        fill: false,
                        tension: 0.4
                    }
                ]
            },
            options: {
                ...ChartConfig.defaultOptions,
                plugins: {
                    ...ChartConfig.defaultOptions.plugins,
                    title: {
                        display: true,
                        text: 'Year-Over-Year Comparison'
                    }
                }
            }
        });
    },

    createModelComparisonChart(data) {
        const ctx = document.getElementById('modelComparisonChart').getContext('2d');
        
        if (charts.modelComparisonChart) {
            charts.modelComparisonChart.destroy();
        }
        
        charts.modelComparisonChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.sarima_predictions.map(d => d.Time),
                datasets: [
                    {
                        label: 'SARIMA Predictions',
                        data: data.sarima_predictions.map(d => d.Predicted_Workable),
                        borderColor: CONSTANTS.CHART_COLORS.primary,
                        fill: false,
                        tension: 0.4
                    },
                    {
                        label: 'RF Predictions',
                        data: data.rf_predictions.map(d => d.RF_Prediction),
                        borderColor: CONSTANTS.CHART_COLORS.purple,
                        fill: false,
                        tension: 0.4
                    }
                ]
            },
            options: {
                ...ChartConfig.defaultOptions,
                plugins: {
                    ...ChartConfig.defaultOptions.plugins,
                    title: {
                        display: true,
                        text: 'SARIMA vs Random Forest Comparison'
                    }
                }
            }
        });
    }
};
// Update Functions Module
const UpdateFunctions = {
    updateDashboard(data) {
        if (!DataValidator.validateData(data)) {
            Logger.error('Invalid data structure received');
            return;
        }

        globalData = data;
        currentMetricsData = data.current_day;//-----------------------------------------------------------------------------------------------------------------------------------
        nextDayMetricsData = data.next_day;

        // Update last update time
        const lastUpdateElement = document.getElementById('lastUpdate');
        if (lastUpdateElement) {
            lastUpdateElement.textContent = data.time;
        }

        const multiMetrics = MultiMetricsCalculator.calculateMultiMetrics(data);
        this.currentMultiMetrics = multiMetrics;

        this.updateMultiMetrics(data);

        // Update metrics based on current view
        if (isShowingTomorrow) {
            this.updateTomorrowMetrics();
        } else {
            this.updateTodayMetrics();
        }

        safeExecute(() => {
            ChartCreator.createPredictionChart(data.current_day, data.Ledger_Information);
            ChartCreator.createTomorrowPredictionChart(data.next_day);
            ChartCreator.createBackLogChart(data);
            ChartCreator.createComparisonChart(data.current_day);
            ChartCreator.createModelComparisonChart(data.current_day);
        }, 'Error updating charts');

        // Update processing metrics
        this.updateProcessingMetrics(data);
    },
    updateTodayMetrics() {
        if (!currentMetricsData) {
            Logger.warn('No current metrics data available');
            return;
        }
        try {
            // Calculate metrics
            const dayShiftMetrics = MetricsCalculator.calculateDayShiftMetrics(currentMetricsData);
            const nightShiftMetrics = MetricsCalculator.calculateNightShiftMetrics(currentMetricsData, nextDayMetricsData);
            // Update day shift metrics with multi splits
            this.updateMetricWithMultiSplit(
                document.getElementById('daysSarimaMorning'),
                document.getElementById('daysSarimaMorningMulti'),
                dayShiftMetrics.morning
            );
            this.updateMetricWithMultiSplit(
                document.getElementById('daysSarimaAfternoon'),
                document.getElementById('daysSarimaAfternoonMulti'),
                dayShiftMetrics.afternoon
            );
            this.updateMetricWithMultiSplit(
                document.getElementById('daysSarimaTotal'),
                document.getElementById('daysSarimaTotalMulti'),
                dayShiftMetrics.total
            );
            this.updateMetricWithMultiSplit(
                document.getElementById('dayShiftProgress'),
                document.getElementById('dayShiftProgressMulti'),
                dayShiftMetrics.progress
            );

            this.updateMetricWithMultiSplit(
                document.getElementById('nightsSarimaToday'),
                document.getElementById('nightsSarimaTodayMulti'),
                nightShiftMetrics.tonight
            );
            this.updateMetricWithMultiSplit(
                document.getElementById('nightsSarimaTomorrow'),
                document.getElementById('nightsSarimaTomorrowMulti'),
                nightShiftMetrics.tomorrow
            );
            this.updateMetricWithMultiSplit(
                document.getElementById('nightsSarimaTotal'),
                document.getElementById('nightsSarimaTotalMulti'),
                nightShiftMetrics.total
            );

            // Update other metrics...
            const {
                network_prediction,
                sarima_predictions,
                predictions_no_same_day
            } = currentMetricsData;

            this.updateMetricWithMultiSplit(
                document.getElementById('targetValue'),
                document.getElementById('targetValueMulti'),
                network_prediction
            );

            // Calculate and update difference
            const finalSarima = predictions_no_same_day[23].Predicted_Workable_No_Same_Day;
            const difference = ((finalSarima - network_prediction) / network_prediction) * 100;
            const differenceElement = document.getElementById('differenceValue');
            differenceElement.textContent = `${difference >= 0 ? '+' : ''}${difference.toFixed(2)}%`;
            differenceElement.className = `difference-value ${difference >= 0 ? 'positive' : 'negative'}`;

            // Update multi-unit progress
            this.updateMultiProgress(this.currentMultiMetrics);

        } catch (error) {
            Logger.error('Error updating today metrics:', error);
        }
    },


    updateTomorrowMetrics() {
        if (!nextDayMetricsData) {
            Logger.warn('No next day metrics data available');
            return;
        }

        try {
            // Handle user prediction and difference calculation
            const userPrediction = parseInt(document.getElementById('userPrediction').value);
            if (userPrediction) {
                const finalSarima = nextDayMetricsData.sarima_predictions[23].Predicted_Workable;
                const difference = ((finalSarima - userPrediction) / userPrediction) * 100;
                
                const differenceElement = document.getElementById('tomorrowPredictionDifference');
                differenceElement.textContent = `${difference >= 0 ? '+' : ''}${difference.toFixed(2)}%`;
                differenceElement.className = `metric-value ${difference >= 0 ? 'positive' : 'negative'}`;

                this.updateMetricWithMultiSplit(
                    document.getElementById('userPrediction'),
                    document.getElementById('userPredictionMulti'),
                    userPrediction
                );
            }

            const dayShiftMetrics = MetricsCalculator.calculateTomorrowDayShiftMetrics(nextDayMetricsData);
            
            // Update morning metrics
            this.updateMetricWithMultiSplit(
                document.getElementById('tomorrowDayMorning'),
                document.getElementById('tomorrowDayMorningMulti'),
                dayShiftMetrics.morning
            );

            // Update afternoon metrics
            this.updateMetricWithMultiSplit(
                document.getElementById('tomorrowDayAfternoon'),
                document.getElementById('tomorrowDayAfternoonMulti'),
                dayShiftMetrics.afternoon
            );

            this.updateMetricWithMultiSplit(
                document.getElementById('tomorrowDayTotal'),
                document.getElementById('tomorrowDayTotalMulti'),
                dayShiftMetrics.total
            );

            const nightShiftMetrics = MetricsCalculator.calculateTomorrowNightShiftMetrics(
                nextDayMetricsData,
                currentMetricsData
            );

            this.updateMetricWithMultiSplit(
                document.getElementById('tomorrowNightFirst'),
                document.getElementById('tomorrowNightFirstMulti'),
                nightShiftMetrics.tomorrow
            );

            // Update early morning metrics
            this.updateMetricWithMultiSplit(
                document.getElementById('todayEarlyMorning'),
                document.getElementById('todayEarlyMorningMulti'),
                nightShiftMetrics.today
            );

            this.updateMetricWithMultiSplit(
                document.getElementById('tomorrowNightTotal'),
                document.getElementById('tomorrowNightTotalMulti'),
                nightShiftMetrics.total
            );

        } catch (error) {
            Logger.error('Error updating tomorrow metrics:', error);
        }
    },

    updateAllShiftMetrics(metrics, prefix) {
        Object.entries(metrics).forEach(([key, value]) => {
            const totalElement = document.getElementById(`${prefix}${key.charAt(0).toUpperCase() + key.slice(1)}`);
            const multiElement = document.getElementById(`${prefix}${key.charAt(0).toUpperCase() + key.slice(1)}Multi`);
            
            if (totalElement && multiElement) {
                this.updateMetricWithMultiSplit(totalElement, multiElement, value);
            }
        });
    },

    updateMetricWithMultiSplit(totalElement, multiElement, value) {
        if (!this.currentMultiMetrics || !totalElement || !multiElement) return;

        const totalValue = Math.round(value);
        const multiValue = MultiMetricsCalculator.calculateMultiSplit(
            value,
            this.currentMultiMetrics.avgIPTM
        );

        totalElement.textContent = totalValue.toLocaleString();
        multiElement.textContent = `(${multiValue.toLocaleString()})`;

        // Add visual indication if multi split is off target
        const actualPercentage = (multiValue / totalValue) * 100;
        const deviation = Math.abs(actualPercentage - this.currentMultiMetrics.avgIPTM);
        
        multiElement.className = 'multi-split' + 
            (deviation > 10 ? ' critical' : 
             deviation > 5 ? ' warning' : '');
    },
    

    updateMultiProgress(metrics) {
        if (!metrics) return;
    
        // Update existing display
        const targetElement = document.getElementById('targetMultiPercentage');
        const currentElement = document.getElementById('currentMultiPercentage');
        
        if (targetElement) {
            targetElement.textContent = `${metrics.avgIPTM.toFixed(1)}%`;
        }
        
        if (currentElement) {
            currentElement.textContent = `${metrics.currentMultiRate.toFixed(1)}%`;
            currentElement.className = metrics.isOnTarget ? 'on-target' : 'off-target';
        }
    
        // Update progress bars
        const progressBar = document.getElementById('multiProgressBar');
        if (progressBar) {
            const fill = progressBar.querySelector('.progress-fill');
            if (fill) {
                const percentage = (metrics.currentMultiRate / metrics.targetMultiRate) * 100;
                fill.style.width = `${Math.min(percentage, 100)}%`;
                fill.style.backgroundColor = percentage < 90 ? '#e74c3c' : 
                                           percentage < 95 ? '#f1c40f' : 
                                           '#2ecc71';
            }
        }
    },
    

    updateMultiMetrics(data) {
        if (!data || !data.Ledger_Information) return;
        
        const multiMetrics = this.currentMultiMetrics;
        if (!multiMetrics) return;
        
        // Update display elements
        document.getElementById('targetMultiPercentage').textContent = 
            `${multiMetrics.targetMultiRate.toFixed(1)}%`;
        document.getElementById('currentMultiPercentage').textContent = 
            `${multiMetrics.currentMultiRate.toFixed(1)}%`;
        
        this.updateMultiProgress(multiMetrics);
    },

    updateMultiMetrics(data) {
        if (!data || !data.Ledger_Information) return;
        
        const multiMetrics = this.currentMultiMetrics;
        if (!multiMetrics) return;
        
        // Get current hour for progress calculation
        const currentHour = new Date().getHours();
        const dayProgress = (currentHour / 24) * 100;
        
        // Get metrics from Ledger Information
        const metrics = data.Ledger_Information.metrics;
        
        // Add debugging logs
        console.log('Full metrics:', metrics);
        console.log('MSSF array:', metrics.MSSF);
        
        // Get last non-zero or non-null MSSF value
        const currentMSSF = metrics.MSSF
            .filter(value => value !== null && value !== undefined && value > 0)
            .pop() || 0;
        
        console.log('Selected MSSF value:', currentMSSF);
        
        const currentIPTM = metrics.IPTM
            .filter(value => value !== null && value !== undefined && value > 0)
            .pop() || 0;
            
        const multiIPT = metrics.IPT[0] || 0;
        
        // Update display elements
        document.getElementById('currentIPTM').textContent = `${currentIPTM.toFixed(1)}%`;
        document.getElementById('multiIPT').textContent = multiIPT.toLocaleString();
        document.getElementById('currentMSSF').textContent = `${currentMSSF.toFixed(1)}%`;
        
        // Update progress bars
        const dayProgressFill = document.getElementById('dayProgressFill');
        const multiProgressFill = document.getElementById('multiProgressFill');
        
        if (dayProgressFill) {
            dayProgressFill.style.width = `${dayProgress}%`;
            document.getElementById('dayProgressValue').textContent = `${dayProgress.toFixed(1)}%`;
        }
        
        if (multiProgressFill) {
            multiProgressFill.style.width = `${currentMSSF}%`;
            document.getElementById('multiProgressValue').textContent = `${currentMSSF.toFixed(1)}%`;
        }
        
        // Update progress status
        const progressStatus = document.getElementById('progressStatus');
        if (progressStatus) {
            if (currentMSSF > dayProgress) {
                progressStatus.className = 'progress-status ahead';
                progressStatus.textContent = `Ahead by ${(currentMSSF - dayProgress).toFixed(1)}%`;
            } else {
                progressStatus.className = 'progress-status behind';
                progressStatus.textContent = `Behind by ${(dayProgress - currentMSSF).toFixed(1)}%`;
            }
        }
        
        // Update existing multi progress display
        this.updateMultiProgress(multiMetrics);
    },    
    
    updateProcessingMetrics(data) {
        const processingRate = MetricsCalculator.calculateProcessingRate(data);
        const processingRateElement = document.getElementById('processingRate');
        const processingRateMultiElement = document.getElementById('processingRateMulti');
        
        if (processingRateElement) {
            const rateDisplay = processingRate > 0 
                ? `<span class="metric-label">Over Processing by:</span>
                   <span class="metric-value trend-number positive">${processingRate.toLocaleString()}</span>`
                : `<span class="metric-label">Under Processing by:</span>
                   <span class="metric-value trend-number negative">${Math.abs(processingRate).toLocaleString()}</span>`;
            
            processingRateElement.innerHTML = rateDisplay;

            if (processingRateMultiElement && this.currentMultiMetrics) {
                const multiRate = MultiMetricsCalculator.calculateMultiSplit(
                    processingRate,
                    this.currentMultiMetrics.avgIPTM
                );
                processingRateMultiElement.textContent = `(${multiRate.toLocaleString()})`;
            }
        }
    }
};

const TimerManager = {
    init() {
        this.timerFill = document.getElementById('timerFill');
        this.countdownDisplay = document.getElementById('countdown');
        
        this.updateTimer();
        this.timerInterval = setInterval(() => this.updateTimer(), 1000);
        this.startTimer();
    },

    updateTimer() {
        const now = new Date();
        const nextUpdate = this.getNextUpdate();
        const timeLeft = nextUpdate - now;
        
        // Calculate minutes and seconds
        const minutes = Math.floor(timeLeft / 60000);
        const seconds = Math.floor((timeLeft % 60000) / 1000);
        
        // Calculate progress percentage (0-100%)
        const progress = this.calculateProgress(now);
        
        // Update countdown display
        if (this.countdownDisplay) {
            this.countdownDisplay.textContent = 
                `${minutes}:${seconds.toString().padStart(2, '0')}`;
        }
        
        // Update progress bar
        if (this.timerFill) {
            requestAnimationFrame(() => {
                this.timerFill.style.width = `${progress}%`;
                this.updateProgressColor(progress);
            });
        }
    },

    calculateProgress(now) {
        const minutes = now.getMinutes();
        const seconds = now.getSeconds();
        
        // Calculate progress within the current hour (0-60 minutes)
        const minutesPassed = (minutes + (seconds / 60)) % 60;
        
        // Calculate progress (100% at :15, 0% just before :15)
        let progress;
        if (minutesPassed < 15) {
            // From :00 to :15, progress goes from 25% to 0%
            progress = 25 - (minutesPassed / 60) * 100;
        } else {
            // From :15 to :00, progress goes from 100% to 25%
            progress = 100 - ((minutesPassed - 15) / 45) * 75;
        }

        return progress;
    },

    updateProgressColor(progress) {
        if (!this.timerFill) return;
        
        let color;
        if (progress < 25) {
            color = 'var(--error-color)';
        } else if (progress < 50) {
            color = 'var(--warning-color)';
        } else {
            color = 'var(--primary-color)';
        }
        
        this.timerFill.style.backgroundColor = color;
    },

    getNextUpdate() {
        const now = new Date();
        const nextUpdate = new Date(now);

        nextUpdate.setMinutes(15);
        nextUpdate.setSeconds(0);
        nextUpdate.setMilliseconds(0);

        if (now >= nextUpdate) {
            nextUpdate.setHours(nextUpdate.getHours() + 1);
        }

        return nextUpdate;
    },

    startTimer() {
        const now = new Date();
        const nextUpdate = this.getNextUpdate();
        const delay = nextUpdate - now;

        setTimeout(() => {
            // Trigger data refresh
            DataProcessor.fetchData();
            
            // Restart timer
            this.startTimer();
        }, delay);
    }
};



const ThemeManager = {
    init() {
        this.themeToggleBtn = document.getElementById('themeToggle');
        this.themeToggleIcon = this.themeToggleBtn.querySelector('.theme-toggle-icon');
        
        // Set initial theme
        const savedTheme = localStorage.getItem('theme') || 'light';
        this.setTheme(savedTheme);
        
        // Add event listener
        this.themeToggleBtn.addEventListener('click', () => this.toggleTheme());
    },

    setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        this.themeToggleIcon.textContent = theme === 'dark' ? '☀️' : '🌙';
        localStorage.setItem('theme', theme);
    },

    toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        this.setTheme(newTheme);
    }
};

class DataCollector {
    constructor() {
        this.API_ENDPOINT = 'https://xuc6hrpmlf.execute-api.us-west-1.amazonaws.com/prod/collect';
        this.WEBSITE_ORIGIN = 'http://ledger-prediction-charting-website.s3-website-us-west-1.amazonaws.com';
        this.MIDWAY_AUTH_URL = 'https://midway-auth.amazon.com';
        this.initializeElements();
        this.setupEventListeners();
    }

    initializeElements() {
        this.elements = {
            collectButton: document.getElementById('collectDataBtn'),
            statusDisplay: document.getElementById('collectionStatus')
        };
        console.log('Elements initialized:', this.elements);
    }

    setupEventListeners() {
        if (this.elements.collectButton) {
            this.elements.collectButton.addEventListener('click', () => this.handleDataCollection());
            console.log('Event listener attached to button');
        } else {
            console.error('Button element not found during setup');
        }
    }

    updateStatus(message, isError = false) {
        if (this.elements.statusDisplay) {
            this.elements.statusDisplay.textContent = message;
            this.elements.statusDisplay.className = `collection-status ${isError ? 'error' : 'success'}`;
            console.log('Status updated:', message);
        }
    }

    createMidwayPrompt() {
        const existingPrompt = document.getElementById('midwayPrompt');
        if (existingPrompt) existingPrompt.remove();

        const prompt = document.createElement('div');
        prompt.id = 'midwayPrompt';
        prompt.className = 'midway-prompt';
        prompt.innerHTML = `
            <div class="midway-prompt-content">
                <h3>Midway Authentication Required</h3>
                <p>Please authenticate with Midway to continue.</p>
                <div class="midway-buttons">
                    <button onclick="window.openMidwayAuth()" class="midway-auth-btn">
                        Login with Midway
                    </button>
                    <button onclick="window.openMwinit()" class="midway-mwinit-btn">
                        Run mwinit
                    </button>
                </div>
                <p class="midway-note">After authenticating, refresh this page and try again.</p>
                <button onclick="window.location.reload()" class="refresh-btn">
                    Refresh Page
                </button>
            </div>
        `;

        document.body.appendChild(prompt);
    }

    async checkMidwayAuth() {
        try {
            // Instead of pinging Midway directly, we'll try our API endpoint
            const response = await fetch(this.API_ENDPOINT, {
                method: 'OPTIONS',
                credentials: 'include',
                headers: {
                    'Origin': this.WEBSITE_ORIGIN
                }
            });
            
            // If we get a 401, we need authentication
            if (response.status === 401) {
                return false;
            }
            
            // Any successful response means we're authenticated
            return response.ok;
            
        } catch (error) {
            console.error('Auth check failed:', error);
            return false;
        }
    }

    async checkCORSAndAuth() {
        try {
            const response = await fetch(this.API_ENDPOINT, {
                method: 'OPTIONS',
                credentials: 'include',
                headers: {
                    'Origin': this.WEBSITE_ORIGIN,
                    'Access-Control-Request-Method': 'POST',
                    'Access-Control-Request-Headers': 'content-type,origin'
                }
            });
            console.log('CORS check response:', response);
            return response.ok;
        } catch (error) {
            console.error('CORS check failed:', error);
            this.updateStatus('Unable to connect to the API. Please check your network connection.', true);
            return false;
        }
    }

    async handleDataCollection() {
        try {
            this.elements.collectButton.disabled = true;
            this.updateStatus('Checking authentication...');

            // First try OPTIONS request
            const optionsResponse = await fetch(this.API_ENDPOINT, {
                method: 'OPTIONS',
                headers: {
                    'Origin': this.WEBSITE_ORIGIN,
                    'Access-Control-Request-Method': 'POST',
                    'Access-Control-Request-Headers': 'Content-Type,Origin,Cookie'
                }
            });

            console.log('OPTIONS response:', optionsResponse);

            if (!optionsResponse.ok) {
                throw new Error('CORS preflight failed');
            }

            // Then make the actual request
            const response = await fetch(this.API_ENDPOINT, {
                method: 'POST',
                credentials: 'include',
                headers: {
                    'Content-Type': 'application/json',
                    'Origin': this.WEBSITE_ORIGIN
                }
            });

            console.log('POST response:', response);

            if (response.status === 401) {
                // Handle authentication required
                this.updateStatus('Authentication required. Redirecting to Midway...');
                window.location.href = `https://midway-auth.amazon.com/login?redirect_uri=${encodeURIComponent(window.location.href)}`;
                return;
            }

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || 'Request failed');
            }

            const data = await response.json();
            this.updateStatus(`Successfully collected ${data.combine_records} combine records and ${data.granular_records} granular records.`);
            
            if (typeof DataProcessor !== 'undefined' && DataProcessor.fetchData) {
                DataProcessor.fetchData();
            }

        } catch (error) {
            console.error('Collection error:', error);
            this.updateStatus(`Error: ${error.message}`, true);
        } finally {
            this.elements.collectButton.disabled = false;
        }
    }
    async checkMidwayAuth() {
        try {
            const response = await fetch(`${this.MIDWAY_AUTH_URL}/ping`, {
                credentials: 'include'
            });
            return response.ok;
        } catch (error) {
            console.error('Midway auth check failed:', error);
            return false;
        }
    }

    // Handle return from Midway
    handleAuthReturn() {
        const returnUrl = localStorage.getItem('returnUrl');
        if (returnUrl) {
            localStorage.removeItem('returnUrl');
            this.handleDataCollection();
        }
    }

    async collectData() {
        try {
            if (!this.elements.collectButton || !this.elements.statusDisplay) {
                console.error('Required elements not found');
                return;
            }

            this.elements.collectButton.disabled = true;
            this.updateStatus('Checking connection...');

            const corsOk = await this.checkCORSAndAuth();
            if (!corsOk) {
                this.updateStatus('Unable to access the API. Please check your network connection.', true);
                return;
            }

            this.updateStatus('Collecting data...');

            const response = await fetch(this.API_ENDPOINT, {
                method: 'POST',
                credentials: 'include',
                headers: {
                    'Content-Type': 'application/json',
                    'Origin': this.WEBSITE_ORIGIN
                }
            });

            console.log('API Response:', response);

            const data = await response.json();
            console.log('Response data:', data);

            if (!response.ok) {
                if (response.status === 401) {
                    this.createMidwayPrompt();
                    return;
                }
                throw new Error(data.message || 'Failed to collect data');
            }

            this.updateStatus(`Successfully collected ${data.combine_records} combine records and ${data.granular_records} granular records.`);
            
            if (typeof DataProcessor !== 'undefined' && DataProcessor.fetchData) {
                DataProcessor.fetchData();
            }

        } catch (error) {
            console.error('Collection error:', error);
            this.updateStatus(`Error: ${error.message}`, true);
        } finally {
            if (this.elements.collectButton) {
                this.elements.collectButton.disabled = false;
            }
        }
    }

    async testCORS() {
        try {
            this.updateStatus('Testing CORS configuration...');
            
            const optionsResponse = await fetch(this.API_ENDPOINT, {
                method: 'OPTIONS',
                headers: {
                    'Origin': this.WEBSITE_ORIGIN,
                    'Access-Control-Request-Method': 'POST',
                    'Access-Control-Request-Headers': 'content-type,origin'
                }
            });

            console.log('OPTIONS response:', {
                status: optionsResponse.status,
                headers: {
                    allowOrigin: optionsResponse.headers.get('Access-Control-Allow-Origin'),
                    allowMethods: optionsResponse.headers.get('Access-Control-Allow-Methods'),
                    allowCredentials: optionsResponse.headers.get('Access-Control-Allow-Credentials')
                }
            });

            const postResponse = await fetch(this.API_ENDPOINT, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Origin': this.WEBSITE_ORIGIN
                },
                credentials: 'include',
                body: JSON.stringify({ test: true })
            });

            console.log('POST response:', postResponse);
            const data = await postResponse.json();
            console.log('POST response data:', data);

            this.updateStatus('CORS test completed');
            return true;
        } catch (error) {
            console.error('CORS test failed:', error);
            this.updateStatus(`CORS test failed: ${error.message}`, true);
            return false;
        }
    }
}
document.addEventListener('DOMContentLoaded', () => {
    const collector = new DataCollector();
    // Check if we're returning from Midway
    if (window.location.href.includes('midway-auth')) {
        collector.handleAuthReturn();
    }
});

// Global helper functions
window.openMidwayAuth = function() {
    const returnUrl = encodeURIComponent(window.location.href);
    const midwayUrl = `https://midway-auth.amazon.com/login?redirect_uri=${returnUrl}`;
    // Open Midway login in a new window
    const authWindow = window.open(midwayUrl, 'MidwayAuth', 
        'width=800,height=600,status=0,toolbar=0');
    
    // Check if window was successfully opened
    if (authWindow) {
        // Poll for window close
        const checkWindow = setInterval(() => {
            if (authWindow.closed) {
                clearInterval(checkWindow);
                window.location.reload(); // Refresh the page after auth
            }
        }, 500);
    } else {
        alert('Please allow popups for this site and try again.');
    }
};

window.openMwinit = function() {
    const modal = document.createElement('div');
    modal.className = 'midway-modal';
    modal.innerHTML = `
        <div class="midway-modal-content">
            <h3>Run mwinit Command</h3>
            <p>Open your terminal and run:</p>
            <code>mwinit -o</code>
            <p>After running the command, click the refresh button below.</p>
            <button onclick="window.location.reload()" class="refresh-btn">
                Refresh Page
            </button>
            <button onclick="this.parentElement.parentElement.remove()" class="close-btn">
                Close
            </button>
        </div>
    `;
    document.body.appendChild(modal);
};

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing DataCollector...');
    window.dataCollector = new DataCollector();
});

// Debug helper
window.testCollector = async function() {
    const collector = new DataCollector();
    console.log('Testing CORS...');
    await collector.testCORS();
    console.log('Testing Midway auth...');
    const isAuth = await collector.checkMidwayAuth();
    console.log('Midway auth status:', isAuth);
};


async function testComplete() {
    const collector = new DataCollector();
    
    console.log('1. Checking Midway authentication...');
    const isAuthenticated = await collector.checkMidwayAuth();
    console.log('Midway auth status:', isAuthenticated);
    
    if (!isAuthenticated) {
        console.log('Please authenticate with Midway first');
        return;
    }
    
    console.log('2. Testing CORS...');
    await collector.testCORS();
}


// Initialize the collector when the document is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing DataCollector...'); // Debug log
    const collector = new DataCollector();
});

// Add a global debug function
window.checkCollectorStatus = function() {
    const button = document.getElementById('collectDataBtn');
    const status = document.getElementById('collectionStatus');
    console.log('Collector Elements Status:', {
        button: {
            exists: !!button,
            disabled: button ? button.disabled : 'N/A',
            hasClickListener: button ? !!button.onclick : 'N/A'
        },
        status: {
            exists: !!status,
            content: status ? status.textContent : 'N/A'
        }
    });
};


// File Upload Module
class FileUploader {
    constructor() {
        this.initializeAWS();
        this.initializeElements();
        this.setupEventListeners();
    }

    initializeAWS() {
        AWS.config.region = CONSTANTS.AWS_REGION;
        AWS.config.credentials = new AWS.CognitoIdentityCredentials({
            IdentityPoolId: CONSTANTS.IDENTITY_POOL_ID
        });

        this.s3 = new AWS.S3({
            params: {
                Bucket: CONSTANTS.BUCKET_NAME
            }
        });
    }

    initializeElements() {
        this.elements = {
            combineInput: document.getElementById('combineFile'),
            granularInput: document.getElementById('granularFile'),
            uploadButton: document.getElementById('uploadButton'),
            uploadMessage: document.getElementById('uploadMessage')
        };

        if (!this.validateElements()) {
            Logger.error('Failed to initialize file upload elements');
        }
    }

    validateElements() {
        return Object.entries(this.elements).every(([key, element]) => {
            if (!element) {
                Logger.error(`Missing element: ${key}`);
                return false;
            }
            return true;
        });
    }

    setupEventListeners() {
        if (this.elements.uploadButton) {
            this.elements.uploadButton.addEventListener('click', () => this.handleUpload());
        }
    }

    showMessage(message, isError = false) {
        if (this.elements.uploadMessage) {
            this.elements.uploadMessage.textContent = message;
            this.elements.uploadMessage.className = `upload-message ${isError ? 'error' : 'success'}`;
            this.elements.uploadMessage.style.display = 'block';
            
            setTimeout(() => {
                this.elements.uploadMessage.style.display = 'none';
            }, 5000);
        }
    }

    async validateFile(file, expectedHeader) {
        if (!file || !file.name.endsWith('.csv')) {
            return false;
        }

        try {
            const text = await file.text();
            const firstLine = text.split('\n')[0].trim();
            return firstLine.startsWith(expectedHeader);
        } catch (error) {
            Logger.error('File validation error:', error);
            return false;
        }
    }

    async uploadFileToS3(file, key) {
        return new Promise((resolve, reject) => {
            const params = {
                Key: `userloads/${key}`,
                Body: file,
                ContentType: 'text/csv'
            };

            this.s3.upload(params, (err, data) => {
                if (err) {
                    Logger.error('Upload error:', err);
                    reject(err);
                } else {
                    resolve(data);
                }
            });
        });
    }

    async handleUpload() {
        const combineFile = this.elements.combineInput.files[0];
        const granularFile = this.elements.granularInput.files[0];

        if (!combineFile || !granularFile) {
            this.showMessage('Please select both files', true);
            return;
        }

        const isCombineValid = await this.validateFile(combineFile, 'Time');
        const isGranularValid = await this.validateFile(granularFile, 'Hour');

        if (!isCombineValid || !isGranularValid) {
            this.showMessage('Invalid file format. Please check headers.', true);
            return;
        }

        this.elements.uploadButton.disabled = true;
        
        try {
            await Promise.all([
                this.uploadFileToS3(combineFile, 'combine.csv'),
                this.uploadFileToS3(granularFile, 'granular.csv')
            ]);

            this.showMessage('Files uploaded successfully!');
            this.elements.combineInput.value = '';
            this.elements.granularInput.value = '';
        } catch (error) {
            this.showMessage(`Upload failed: ${error.message}`, true);
        } finally {
            this.elements.uploadButton.disabled = false;
        }
    }
}

// Event Handlers
const EventHandlers = {
    setupInputHandlers() {
        const userPredictionInput = document.getElementById('userPrediction');
        if (userPredictionInput) {
            userPredictionInput.addEventListener('input', function(e) {
                this.value = this.value.replace(/[^0-9]/g, '');
                if (this.value) {
                    UpdateFunctions.updateTomorrowMetrics();
                }
            });
        }
    }
};



// Application Initialization
function initializeApplication() {
    UIHelpers.setupMetricsToggle();

    UIHelpers.createRefreshButton();
    
    const uploader = new FileUploader();
    const collector = new DataCollector();
    
    // Check if collector was initialized properly
    if (!collector.elements.collectButton || !collector.elements.statusDisplay) {
        console.error('Failed to initialize DataCollector - required elements not found');
    }    EventHandlers.setupInputHandlers();
    
    DataProcessor.fetchData();

    setInterval(() => DataProcessor.fetchData(), CONSTANTS.REFRESH_INTERVAL);
}

// Start the application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Your existing initialization code
    initializeApplication();
    TimerManager.init();
    ThemeManager.init();
});