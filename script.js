// Revenue Predictor Frontend Script
// Handles form submission, API communication, and GSAP animations

// Supabase Configuration
console.log('Script loaded successfully (sbClient build 20260114)');
const supabaseUrl = 'https://hvboifzfawryabsgaau.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh2Ym9pZnpmYXdyeWFic2dzYWF1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjgyNDY4MDksImV4cCI6MjA4MzgyMjgwOX0.42dgZvia94nPhtYmV5MxHhkk19yzw9BYDyVucq9zDT8';

// Initialize Supabase client
const sbClient = window.supabase.createClient(supabaseUrl, SUPABASE_ANON_KEY);

// API endpoint configuration
// Using Render deployment URL for production
const API_BASE_URL = 'https://revenue-predictor-5lqt.onrender.com';
const PREDICT_ENDPOINT = `${API_BASE_URL}/predict`;
const REPORT_ENDPOINT = `${API_BASE_URL}/report`;
const HISTORY_ENDPOINT = `${API_BASE_URL}/history`;

// DOM element references - with null checks to prevent undefined errors
const form = document.getElementById('prediction-form');
const resultsSection = document.getElementById('results-section');
const predictedRevenueEl = document.getElementById('predicted-revenue');
const predictBtn = document.getElementById('predict-btn');
const downloadReportBtn = document.getElementById('downloadReport');
const backBtn = document.getElementById('back-btn');
const tryPredictorBtn = document.getElementById('try-predictor-btn');
const loadingEl = document.getElementById('loading');
const loadingTextEl = document.getElementById('loading-text');
const errorMessageEl = document.getElementById('error-message');

// Authentication DOM elements
const authModal = document.getElementById('auth-modal');
const authModalClose = document.getElementById('auth-modal-close');
const loginForm = document.getElementById('login-form-element');
const signupForm = document.getElementById('signup-form-element');
const loginTab = document.querySelector('[data-tab="login"]');
const signupTab = document.querySelector('[data-tab="signup"]');
const logoutBtn = document.getElementById('logout-btn');
const userEmail = document.getElementById('user-email');
const loginError = document.getElementById('login-error');
const signupError = document.getElementById('signup-error');

// Chart.js instances
let contributionsChart = null;
let revenueChart = null;

// Authentication Functions
async function checkAuth() {
    try {
        const { data: { session } } = await sbClient.auth.getSession();
        return session;
    } catch (error) {
        console.error('Error checking auth:', error);
        return null;
    }
}

async function handleLogin(email, password) {
    try {
        const { data, error } = await sbClient.auth.signInWithPassword({
            email: email,
            password: password
        });
        
        if (error) throw error;
        return data;
    } catch (error) {
        console.error('Login error:', error);
        throw error;
    }
}

async function handleSignup(email, password) {
    try {
        const { data, error } = await sbClient.auth.signUp({
            email: email,
            password: password
        });
        
        if (error) throw error;
        return data;
    } catch (error) {
        console.error('Signup error:', error);
        throw error;
    }
}

async function handleLogout() {
    try {
        const { error } = await sbClient.auth.signOut();
        if (error) throw error;
    } catch (error) {
        console.error('Logout error:', error);
        showError('Failed to logout. Please try again.');
    }
}

function updateUIForAuth(user) {
    if (user) {
        // User is logged in
        if (authModal) authModal.classList.add('hidden');
        if (userEmail) {
            userEmail.textContent = user.email;
            userEmail.classList.remove('hidden');
        }
        if (logoutBtn) {
            logoutBtn.classList.remove('hidden');
        }
        // Show main content
        const mainContent = document.querySelector('.main-content');
        const sidebar = document.querySelector('.sidebar');
        if (mainContent) mainContent.style.display = 'flex';
        if (sidebar) sidebar.style.display = 'flex';
    } else {
        // User is not logged in
        if (authModal) authModal.classList.remove('hidden');
        if (userEmail) {
            userEmail.classList.add('hidden');
        }
        if (logoutBtn) {
            logoutBtn.classList.add('hidden');
        }
        // Hide main content
        const mainContent = document.querySelector('.main-content');
        const sidebar = document.querySelector('.sidebar');
        if (mainContent) mainContent.style.display = 'none';
        if (sidebar) sidebar.style.display = 'none';
    }
}

async function getCurrentUser() {
    const session = await checkAuth();
    return session ? session.user : null;
}

// Helper function to get auth headers for API requests
async function getAuthHeaders() {
    const session = await checkAuth();
    const headers = {
        'Content-Type': 'application/json',
    };
    if (session && session.access_token) {
        headers['Authorization'] = `Bearer ${session.access_token}`;
    }
    return headers;
}

// Initialize slider value displays
// Updates the displayed value next to each slider as user moves it
function initializeSliders() {
    const sliders = document.querySelectorAll('.slider');
    
    sliders.forEach(slider => {
        // Get the corresponding value display element
        const valueEl = document.getElementById(`${slider.id}_value`);
        
        // Update display on input
        slider.addEventListener('input', (e) => {
            updateSliderValue(slider, valueEl);
        });
        
        // Initial value update
        updateSliderValue(slider, valueEl);
    });
}

// Updates the displayed value for a slider with proper formatting
function updateSliderValue(slider, valueEl) {
    let value = parseFloat(slider.value);
    let formattedValue;
    
    // Format based on slider type
    if (slider.id === 'ad_spend') {
        formattedValue = `$${value.toLocaleString('en-US', { maximumFractionDigits: 0 })}`;
    } else if (slider.id === 'website_visits') {
        formattedValue = value.toLocaleString('en-US');
    } else if (slider.id === 'average_product_price') {
        formattedValue = `$${value.toLocaleString('en-US', { maximumFractionDigits: 0 })}`;
    } else if (slider.id === 'location_score') {
        formattedValue = value.toLocaleString('en-US', { maximumFractionDigits: 0 });
    }
    
    valueEl.textContent = formattedValue;
    
    // Auto-calculate What-If scenarios when slider changes
    calculateWhatIfScenarios();
}

// Calculate What-If scenarios automatically
// Uses the optimized /simulate endpoint for batch scenario calculation
async function calculateWhatIfScenarios() {
    // Get current form values
    const currentData = {
        ad_spend: parseFloat(document.getElementById('ad_spend').value),
        website_visits: parseFloat(document.getElementById('website_visits').value),
        average_product_price: parseFloat(document.getElementById('average_product_price').value),
        location_score: parseFloat(document.getElementById('location_score').value)
    };
    
    // Map scenario IDs to API response keys
    const scenarioMapping = {
        'whatif-standard-revenue': 'standard',
        'whatif-double-ad-revenue': 'double_ad_spend',
        'whatif-viral-revenue': 'double_visits'
    };
    
    try {
        // Use the optimized /simulate endpoint for batch calculation
        const headers = await getAuthHeaders();
        const response = await fetch(`${API_BASE_URL}/simulate`, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(currentData)
        });
        
        if (!response.ok) {
            throw new Error('Failed to get scenario predictions');
        }
        
        const data = await response.json();
        
        // Update each card with the calculated revenue
        Object.entries(scenarioMapping).forEach(([elementId, scenarioKey]) => {
            const element = document.getElementById(elementId);
            if (element && data.scenarios && data.scenarios[scenarioKey] !== undefined) {
                const revenue = data.scenarios[scenarioKey];
                const formattedRevenue = revenue.toLocaleString('en-US', {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2
                });
                
                // Animate the number change
                gsap.to(element, {
                    opacity: 0,
                    duration: 0.2,
                    onComplete: () => {
                        element.textContent = formattedRevenue;
                        gsap.to(element, {
                            opacity: 1,
                            duration: 0.3
                            
                        });
                    }
                });
            } else if (element) {
                element.textContent = '-';
            }
        });
    } catch (error) {
        console.error('Error calculating What-If scenarios:', error);
        // Set all to '-' on error
        Object.keys(scenarioMapping).forEach(elementId => {
            const element = document.getElementById(elementId);
            if (element) {
                element.textContent = '-';
            }
        });
    }
}

// Form submission handler
// Prevents default form submission and sends POST request to API
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Disable button and show loading
    predictBtn.disabled = true;
    showLoading();
    hideError();
    
    // Collect form data
    const formData = {
        ad_spend: parseFloat(document.getElementById('ad_spend').value),
        website_visits: parseFloat(document.getElementById('website_visits').value),
        average_product_price: parseFloat(document.getElementById('average_product_price').value),
        location_score: parseFloat(document.getElementById('location_score').value)
    };
    
    try {
        // Send POST request to /predict endpoint
        const headers = await getAuthHeaders();
        const response = await fetch('https://revenue-predictor-5lqt.onrender.com/predict', {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(formData)
        });
        
        // Check if request was successful
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to get prediction');
        }
        
        // Parse response data
        const data = await response.json();
        
        // Hide loading and display results
        hideLoading();
        displayResults(data);
        
    } catch (error) {
        // Handle errors
        console.error('Prediction error:', error);
        hideLoading();
        showError(error.message || 'Failed to connect to the API. Make sure the server is running.');
        predictBtn.disabled = false;
    }
});

// Download Executive Report button handler
// Collects form data and downloads PDF report from /report endpoint
downloadReportBtn.addEventListener('click', async () => {
    // Disable button and show loading
    downloadReportBtn.disabled = true;
    showLoading('Generating PDF report...');
    hideError();
    
    // Collect form data (same as predict function)
    const formData = {
        ad_spend: parseFloat(document.getElementById('ad_spend').value),
        website_visits: parseFloat(document.getElementById('website_visits').value),
        average_product_price: parseFloat(document.getElementById('average_product_price').value),
        location_score: parseFloat(document.getElementById('location_score').value)
    };
    
    try {
        // Send POST request to /report endpoint
        const headers = await getAuthHeaders();
        const response = await fetch(REPORT_ENDPOINT, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(formData)
        });
        
        // Check if request was successful
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to generate report');
        }
        
        // Get the PDF content as blob
        const blob = await response.blob();
        
        // Create a download link and trigger download
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'revenue_report.pdf';  // Set filename
        document.body.appendChild(a);
        a.click();
        
        // Clean up
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        // Hide loading and re-enable button
        hideLoading();
        downloadReportBtn.disabled = false;
        
    } catch (error) {
        // Handle errors
        console.error('Report download error:', error);
        hideLoading();
        showError(error.message || 'Failed to download report. Make sure the server is running.');
        downloadReportBtn.disabled = false;
    }
});

// Run Growth Scenarios button handler - only if element exists
// Calculates 3 different predictions: Current, Double Marketing, and Viral Growth
const runScenariosBtn = document.getElementById('run-scenarios-btn');
if (runScenariosBtn) {
    runScenariosBtn.addEventListener('click', async () => {
    // Disable button and show loading
    runScenariosBtn.disabled = true;
    showLoading('Calculating growth scenarios...');
    hideError();
    
    // Collect current form data
    const currentData = {
        ad_spend: parseFloat(document.getElementById('ad_spend').value),
        website_visits: parseFloat(document.getElementById('website_visits').value),
        average_product_price: parseFloat(document.getElementById('average_product_price').value),
        location_score: parseFloat(document.getElementById('location_score').value)
    };
    
    // Define 3 scenarios
    const scenarios = [
        {
            name: 'Current',
            description: 'Your current metrics',
            data: { ...currentData }
        },
        {
            name: 'Double Marketing',
            description: '+100% Ad Spend',
            data: {
                ...currentData,
                ad_spend: currentData.ad_spend * 2
            }
        },
        {
            name: 'Viral Growth',
            description: '+100% Website Visits',
            data: {
                ...currentData,
                website_visits: currentData.website_visits * 2
            }
        }
    ];
    
    try {
        // Make 3 API calls in parallel for all scenarios
        const scenarioPromises = scenarios.map(scenario => 
            getAuthHeaders().then(headers => {
                return fetch(PREDICT_ENDPOINT, {
                    method: 'POST',
                    headers: headers,
                    body: JSON.stringify(scenario.data)
                });
            }).then(response => {
                if (!response.ok) {
                    throw new Error(`Failed to get prediction for ${scenario.name}`);
                }
                return response.json();
            }).then(data => ({
                ...scenario,
                predicted_revenue: data.predicted_revenue
            }))
        );
        
        // Wait for all predictions to complete
        const results = await Promise.all(scenarioPromises);
        
        // Hide loading
        hideLoading();
        
        // Show results section if hidden
        if (resultsSection && resultsSection.classList.contains('hidden')) {
            const emptyState = document.getElementById('empty-state');
            if (emptyState) {
                emptyState.classList.add('hidden');
            }
            resultsSection.classList.remove('hidden');
            
            // Animate cards in
            const cards = resultsSection.querySelectorAll('.dashboard-card');
            cards.forEach((card, index) => {
                gsap.fromTo(card,
                    { opacity: 0, y: 20 },
                    { opacity: 1, y: 0, duration: 0.5, delay: index * 0.1, ease: 'power2.out' }
                );
            });
        }
        
    } catch (error) {
        // Handle errors
        console.error('Scenarios error:', error);
        hideLoading();
        showError(error.message || 'Failed to calculate scenarios. Make sure the server is running.');
    } finally {
        if (runScenariosBtn) {
            runScenariosBtn.disabled = false;
        }
    }
    });
}

// Display growth scenarios as comparison cards - removed (not in current HTML structure)
// This functionality is now handled by the What-If scenarios cards in the dashboard

// Display prediction results with animations
function displayResults(data) {
    // Show results section if hidden
    if (resultsSection && resultsSection.classList.contains('hidden')) {
        const emptyState = document.getElementById('empty-state');
        if (emptyState) {
            emptyState.classList.add('hidden');
        }
        resultsSection.classList.remove('hidden');
        
        // Animate cards in
        const cards = resultsSection.querySelectorAll('.dashboard-card');
        cards.forEach((card, index) => {
            gsap.fromTo(card, 
                { opacity: 0, y: 20 },
                { opacity: 1, y: 0, duration: 0.5, delay: index * 0.1, ease: 'power2.out' }
            );
        });
    }
    
    // Animate revenue counter from 0 to predicted value
    // Ensure predicted_revenue is a valid number
    const revenue = parseFloat(data.predicted_revenue) || 0;
    if (isNaN(revenue) || revenue < 0) {
        console.warn('Invalid predicted_revenue value:', data.predicted_revenue);
    }
    animateRevenueCounter(revenue);
    
    // Calculate target goal (120% of predicted revenue as a benchmark)
    const targetGoal = revenue * 1.2;
    
    // Update revenue insight chart
    updateChart(revenue, targetGoal);
    
    // Create and animate bar chart for feature contributions
    createContributionsChart(data.explanation);
    
    // Load prediction history
    loadHistory();
}

// Animate revenue counter using GSAP
// Counts from 0 to the predicted revenue value with smooth animation
function animateRevenueCounter(targetValue) {
    // Reset to 0
    predictedRevenueEl.textContent = '0';
    
    // Create a GSAP timeline for the counting animation
    gsap.to({ value: 0 }, {
        value: targetValue,
        duration: 2,
        ease: 'power2.out',
        onUpdate: function() {
            // Format the number as dollars with commas and 2 decimal places
            const value = this.targets()[0].value;
            const formatted = value.toLocaleString('en-US', {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            });
            predictedRevenueEl.textContent = formatted;
        },
        // Add a slight scale animation for emphasis
        onStart: () => {
            gsap.to(predictedRevenueEl, {
                scale: 1.2,
                duration: 0.3,
                yoyo: true,
                repeat: 1,
                ease: 'power2.inOut'
            });
        }
    });
}

// Update Revenue Insight Chart
// Creates a bar chart comparing Predicted Revenue vs Target Goal
function updateChart(predicted, goal) {
    // Destroy existing chart if it exists
    if (revenueChart) {
        revenueChart.destroy();
    }
    
    // Get canvas context - check if element exists
    const chartCanvas = document.getElementById('revenueChart');
    if (!chartCanvas) {
        console.error('Revenue chart canvas not found');
        return;
    }
    const ctx = chartCanvas.getContext('2d');
    
    // Professional color palette: Blues, Teals, and Grays
    const chartColors = {
        blue: {
            primary: '#3b82f6',      // Blue-500
            light: 'rgba(59, 130, 246, 0.8)',
            dark: '#2563eb'          // Blue-600
        },
        teal: {
            primary: '#14b8a6',      // Teal-500
            light: 'rgba(20, 184, 166, 0.8)',
            dark: '#0d9488'          // Teal-600
        },
        gray: {
            primary: '#64748b',      // Slate-500
            light: 'rgba(100, 116, 139, 0.6)',
            dark: '#475569'          // Slate-600
        }
    };
    
    // Create new Chart.js bar chart
    revenueChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Predicted Revenue', 'Target Goal'],
            datasets: [{
                label: 'Revenue (dollars)',
                data: [predicted, goal],
                backgroundColor: [
                    chartColors.blue.light,   // Blue for predicted revenue
                    chartColors.teal.light    // Teal for target goal
                ],
                borderColor: [
                    chartColors.blue.dark,    // Blue border
                    chartColors.teal.dark     // Teal border
                ],
                borderWidth: 1.5,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleFont: {
                        size: 14,
                        weight: '600'
                    },
                    bodyFont: {
                        size: 13
                    },
                    callbacks: {
                        label: function(context) {
                            const value = context.parsed.y;
                            return value.toLocaleString('en-US', {
                                style: 'currency',
                                currency: 'USD',
                                minimumFractionDigits: 0,
                                maximumFractionDigits: 0
                            });
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: '#e2e8f0',
                        lineWidth: 1
                    },
                    ticks: {
                        color: '#64748b',
                        font: {
                            size: 11
                        },
                        callback: function(value) {
                            // Format as dollars with 'k' suffix for readability
                            if (value >= 1000) {
                                return '$' + (value / 1000).toFixed(1) + 'k';
                            } else {
                                return '$' + value.toFixed(0);
                            }
                        }
                    },
                    title: {
                        display: true,
                        text: 'Revenue',
                        color: '#475569',
                        font: {
                            size: 12,
                            weight: '600'
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#64748b',
                        font: {
                            size: 11
                        }
                    }
                }
            },
            // Animation configuration
            animation: {
                duration: 1500,
                easing: 'easeOutQuart',
                onComplete: () => {
                    // Animate bars individually using GSAP after chart renders
                    animateRevenueChartBars();
                }
            }
        }
    });
}

// Animate revenue chart bars individually using GSAP
function animateRevenueChartBars() {
    if (!revenueChart) return;
    
    // Get all bar elements from the chart
    const meta = revenueChart.getDatasetMeta(0);
    const bars = meta.data;
    
    // Animate each bar with stagger effect
    bars.forEach((bar, index) => {
        // Set initial state
        gsap.set(bar, { scaleY: 0, transformOrigin: 'bottom' });
        
        // Animate to full height with delay
        gsap.to(bar, {
            scaleY: 1,
            duration: 0.6,
            delay: index * 0.1,
            ease: 'back.out(1.7)'
        });
    });
}

// Create Chart.js bar chart for feature contributions
function createContributionsChart(explanation) {
    // Destroy existing chart if it exists
    if (contributionsChart) {
        contributionsChart.destroy();
    }
    
    // Extract feature names and values from explanation object
    // Filter out bias_contribution as it's handled separately
    const featureNames = [];
    const featureValues = [];
    
    // Professional color palette: Blues, Teals, and Grays
    const chartColors = [
        { bg: '#3b82f6', border: '#2563eb' },  // Blue-500/600
        { bg: '#14b8a6', border: '#0d9488' },  // Teal-500/600
        { bg: '#64748b', border: '#475569' },  // Slate-500/600
        { bg: '#2563eb', border: '#1d4ed8' },  // Blue-600/700
        { bg: '#0d9488', border: '#0f766e' },  // Teal-600/700
        { bg: '#475569', border: '#334155' }   // Slate-600/700
    ];
    
    // Create color arrays
    const colors = [];
    const borderColors = [];
    
    // Process each feature contribution
    Object.keys(explanation).forEach((key, index) => {
        // Skip bias contribution for the main chart (or include it if desired)
        if (key !== 'bias_contribution') {
            // Format feature name: "ad_spend_contribution" -> "Ad Spend"
            const formattedName = key
                .replace('_contribution', '')
                .split('_')
                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                .join(' ');
            
            featureNames.push(formattedName);
            featureValues.push(explanation[key]);
            
            // Assign colors cycling through the palette
            const colorSet = chartColors[colors.length % chartColors.length];
            colors.push(colorSet.bg);
            borderColors.push(colorSet.border);
        }
    });
    
    // Get canvas context - check if element exists
    const chartCanvas = document.getElementById('contributions-chart');
    if (!chartCanvas) {
        console.error('Contributions chart canvas not found');
        return;
    }
    const ctx = chartCanvas.getContext('2d');
    
    // Create new Chart.js bar chart
    contributionsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: featureNames,
            datasets: [{
                label: 'Contribution to Revenue',
                data: featureValues,
                backgroundColor: colors,
                borderColor: borderColors,
                borderWidth: 1.5,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleFont: {
                        size: 14,
                        weight: '600'
                    },
                    bodyFont: {
                        size: 13
                    },
                    callbacks: {
                        label: function(context) {
                            const value = context.parsed.y;
                            const formatted = value.toLocaleString('en-US', {
                                style: 'currency',
                                currency: 'USD',
                                minimumFractionDigits: 0,
                                maximumFractionDigits: 0
                            });
                            return `${formatted} contribution`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: '#e2e8f0',
                        lineWidth: 1
                    },
                    ticks: {
                        color: '#64748b',
                        font: {
                            size: 11
                        },
                        callback: function(value) {
                            const formatted = value.toLocaleString('en-US', {
                                style: 'currency',
                                currency: 'USD',
                                minimumFractionDigits: 0,
                                maximumFractionDigits: 0
                            });
                            return formatted;
                        }
                    },
                    title: {
                        display: true,
                        text: 'Contribution',
                        color: '#475569',
                        font: {
                            size: 12,
                            weight: '600'
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#64748b',
                        font: {
                            size: 11
                        }
                    }
                }
            },
            // Animation configuration
            animation: {
                duration: 1500,
                easing: 'easeOutQuart',
                onComplete: () => {
                    // Animate bars individually using GSAP after chart renders
                    animateChartBars();
                }
            }
        }
    });
}

// Animate chart bars individually using GSAP
// Creates a staggered animation effect for visual appeal
function animateChartBars() {
    if (!contributionsChart) return;
    
    // Get all bar elements from the chart
    const meta = contributionsChart.getDatasetMeta(0);
    const bars = meta.data;
    
    // Animate each bar with stagger effect
    bars.forEach((bar, index) => {
        // Set initial state
        gsap.set(bar, { scaleY: 0, transformOrigin: 'bottom' });
        
        // Animate to full height with delay
        gsap.to(bar, {
            scaleY: 1,
            duration: 0.6,
            delay: index * 0.1,
            ease: 'back.out(1.7)'
        });
    });
}

// Load prediction history from /history endpoint
// Fetches data and populates the history table
async function loadHistory() {
    try {
        // Fetch history data from API
        const headers = await getAuthHeaders();
        const response = await fetch(HISTORY_ENDPOINT, {
            method: 'GET',
            headers: headers
        });
        
        // Check if request was successful
        if (!response.ok) {
            // If Supabase is not configured, silently fail (don't show error)
            if (response.status === 503) {
                console.log('History feature not available (Supabase not configured)');
                return;
            }
            throw new Error('Failed to load history');
        }
        
        // Parse response data
        const historyData = await response.json();
        
        // Get table body element
        const tbody = document.getElementById('history-tbody');
        
        // Clear existing rows
        tbody.innerHTML = '';
        
        // Check if we have data
        if (!historyData || historyData.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" style="text-align: center; padding: 20px; color: #666;">No prediction history available</td></tr>';
            return;
        }
        
        // Loop through history data and create table rows
        historyData.forEach((record) => {
            const row = document.createElement('tr');
            
            // Format date (convert ISO string to readable format)
            const date = new Date(record.created_at);
            const formattedDate = date.toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
            
            // Format ad spend
            const adSpend = typeof record.ad_spend === 'number' 
                ? `$${record.ad_spend.toLocaleString('en-US', { maximumFractionDigits: 0 })}` 
                : '$0';
            
            // Format website visits
            const visits = typeof record.website_visits === 'number' 
                ? record.website_visits.toLocaleString('en-US', { maximumFractionDigits: 0 })
                : '0';
            
            // Format predicted revenue
            const revenue = typeof record.predicted_revenue === 'number'
                ? `$${record.predicted_revenue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
                : '$0.00';
            
            // Create table cells
            row.innerHTML = `
                <td>${formattedDate}</td>
                <td>${adSpend}</td>
                <td>${visits}</td>
                <td>${revenue}</td>
            `;
            
            // Append row to table body
            tbody.appendChild(row);
        });
        
        // Animate table rows with GSAP
        const rows = tbody.querySelectorAll('tr');
        rows.forEach((row, index) => {
            gsap.fromTo(row,
                { opacity: 0, y: 10 },
                { opacity: 1, y: 0, duration: 0.4, delay: index * 0.05, ease: 'power2.out' }
            );
        });
        
    } catch (error) {
        // Handle errors silently (don't show error if Supabase is not configured)
        console.error('Error loading history:', error);
        const tbody = document.getElementById('history-tbody');
        if (tbody) {
            tbody.innerHTML = '<tr><td colspan="4" style="text-align: center; padding: 20px; color: #666;">Unable to load history</td></tr>';
        }
    }
}

// Back button handler - returns to empty state
if (backBtn) {
    backBtn.addEventListener('click', () => {
        // Animate transition back to empty state
        gsap.to(resultsSection, {
            opacity: 0,
            y: 20,
            duration: 0.5,
            ease: 'power2.in',
            onComplete: () => {
                resultsSection.classList.add('hidden');
                
                // Show empty state
                const emptyState = document.getElementById('empty-state');
                if (emptyState) {
                    emptyState.classList.remove('hidden');
                    gsap.fromTo(emptyState,
                        { opacity: 0, y: -20 },
                        { opacity: 1, y: 0, duration: 0.6, ease: 'power2.out' }
                    );
                }
                
                // Re-enable predict button
                if (predictBtn) {
                    predictBtn.disabled = false;
                }
            }
        });
    });
}

// Show loading indicator
function showLoading(message = 'Calculating prediction...') {
    // Update loading text
    if (loadingTextEl) {
        loadingTextEl.textContent = message;
    }
    // Show loading element
    loadingEl.classList.remove('hidden');
    gsap.fromTo(loadingEl,
        { opacity: 0 },
        { opacity: 1, duration: 0.3 }
    );
}

// Hide loading indicator
function hideLoading() {
    gsap.to(loadingEl, {
        opacity: 0,
        duration: 0.3,
        onComplete: () => {
            loadingEl.classList.add('hidden');
        }
    });
}

// Show error message
function showError(message) {
    errorMessageEl.textContent = message;
    errorMessageEl.classList.remove('hidden');
    
    // Animate error message appearance
    gsap.fromTo(errorMessageEl,
        { opacity: 0, y: -20 },
        { opacity: 1, y: 0, duration: 0.4, ease: 'back.out(1.7)' }
    );
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        hideError();
    }, 5000);
}

// Hide error message
function hideError() {
    gsap.to(errorMessageEl, {
        opacity: 0,
        y: -20,
        duration: 0.3,
        onComplete: () => {
            errorMessageEl.classList.add('hidden');
        }
    });
}

// Smooth scroll to input section from hero button
if (tryPredictorBtn) {
    tryPredictorBtn.addEventListener('click', () => {
        // Smooth scroll to input section
        inputSection.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
        });
        
        // Optional: Add a subtle highlight animation on the input section
        gsap.fromTo(inputSection,
            { boxShadow: '0 0 0px rgba(109, 40, 217, 0)' },
            { 
                boxShadow: '0 0 30px rgba(109, 40, 217, 0.5)',
                duration: 0.5,
                yoyo: true,
                repeat: 1
            }
        );
    });
}

// Sidebar navigation handler
document.addEventListener('DOMContentLoaded', () => {
    const navLinks = document.querySelectorAll('.nav-link');
    const pages = document.querySelectorAll('.dashboard-page');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetPage = link.getAttribute('data-page');
            
            // Update active state
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            
            // Show/hide pages
            pages.forEach(page => {
                if (page.id === `${targetPage}-page`) {
                    page.classList.remove('hidden');
                } else {
                    page.classList.add('hidden');
                }
            });
        });
    });
    
    // Load history when History page is accessed
    const historyLink = document.querySelector('[data-page="history"]');
    if (historyLink) {
        historyLink.addEventListener('click', () => {
            setTimeout(() => {
                loadHistory();
            }, 100);
        });
    }
    
    // TEMPORARY: Auth bypassed for testing - REMOVE BEFORE PRODUCTION
    console.log('⚠️ WARNING: Authentication is bypassed for testing');
    updateUIForAuth({ email: 'test@example.com' });
    
    // TEMPORARY: Commented out for testing
    // sbClient.auth.onAuthStateChange((event, session) => {
    //     if (event === 'SIGNED_IN' && session) {
    //         updateUIForAuth(session.user);
    //     } else if (event === 'SIGNED_OUT') {
    //         updateUIForAuth(null);
    //     }
    // });
    
    // Login form handler
    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = document.getElementById('login-email').value;
            const password = document.getElementById('login-password').value;
            
            if (loginError) {
                loginError.classList.add('hidden');
            }
            
            try {
                await handleLogin(email, password);
                // UI will update via auth state change listener
            } catch (error) {
                if (loginError) {
                    loginError.textContent = error.message || 'Login failed. Please check your credentials.';
                    loginError.classList.remove('hidden');
                }
            }
        });
    }
    
    // Signup form handler
    if (signupForm) {
        signupForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = document.getElementById('signup-email').value;
            const password = document.getElementById('signup-password').value;
            const passwordConfirm = document.getElementById('signup-password-confirm').value;
            
            if (signupError) {
                signupError.classList.add('hidden');
            }
            
            if (password !== passwordConfirm) {
                if (signupError) {
                    signupError.textContent = 'Passwords do not match.';
                    signupError.classList.remove('hidden');
                }
                return;
            }
            
            if (password.length < 6) {
                if (signupError) {
                    signupError.textContent = 'Password must be at least 6 characters.';
                    signupError.classList.remove('hidden');
                }
                return;
            }
            
            try {
                await handleSignup(email, password);
                if (signupError) {
                    signupError.textContent = 'Sign up successful! Please check your email to verify your account.';
                    signupError.style.background = '#d1fae5';
                    signupError.style.color = '#065f46';
                    signupError.classList.remove('hidden');
                }
            } catch (error) {
                if (signupError) {
                    signupError.textContent = error.message || 'Sign up failed. Please try again.';
                    signupError.classList.remove('hidden');
                }
            }
        });
    }
    
    // Logout button handler
    if (logoutBtn) {
        logoutBtn.addEventListener('click', async () => {
            await handleLogout();
        });
    }
    
    // Auth tab switching
    const loginTab = document.querySelector('[data-tab="login"]');
    const signupTab = document.querySelector('[data-tab="signup"]');
    if (loginTab && signupTab) {
        const switchTab = (tabName) => {
            const loginFormEl = document.getElementById('login-form');
            const signupFormEl = document.getElementById('signup-form');
            
            if (tabName === 'login') {
                loginTab.classList.add('active');
                signupTab.classList.remove('active');
                if (loginFormEl) loginFormEl.classList.remove('hidden');
                if (signupFormEl) signupFormEl.classList.add('hidden');
            } else {
                signupTab.classList.add('active');
                loginTab.classList.remove('active');
                if (signupFormEl) signupFormEl.classList.remove('hidden');
                if (loginFormEl) loginFormEl.classList.add('hidden');
            }
        };
        
        loginTab.addEventListener('click', () => switchTab('login'));
        signupTab.addEventListener('click', () => switchTab('signup'));
    }
});

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Initialize slider value displays
    initializeSliders();
    
    // Set initial GSAP states for animations
    if (resultsSection) {
        gsap.set(resultsSection, { opacity: 0 });
    }
    if (loadingEl) {
        gsap.set(loadingEl, { opacity: 0 });
    }
    if (errorMessageEl) {
        gsap.set(errorMessageEl, { opacity: 0 });
    }
    
    // Calculate initial What-If scenarios
    calculateWhatIfScenarios();
    
    console.log('Revenue Predictor Frontend initialized');
});

