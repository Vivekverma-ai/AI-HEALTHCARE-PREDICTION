// ==========================================
// Form Submission Handler
// ==========================================

const form = document.getElementById('healthForm');
const submitBtn = document.getElementById('submitBtn');
const loader = document.getElementById('loader');
const formContainer = document.getElementById('formContainer');
const resultContainer = document.getElementById('resultContainer');

form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Clear previous errors
    clearErrors();

    // Get form data
    const formData = {
        age: document.getElementById('age').value,
        gender: document.getElementById('gender').value,
        bmi: document.getElementById('bmi').value,
        bp_systolic: document.getElementById('bp_systolic').value,
        bp_diastolic: document.getElementById('bp_diastolic').value,
        blood_sugar: document.getElementById('blood_sugar').value,
        symptoms: document.getElementById('symptoms').value
    };

    // Validate form data
    if (!validateForm(formData)) {
        return;
    }

    // Show loader
    showLoader();

    try {
        // Send request to API
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        const result = await response.json();

        if (response.ok) {
            // Display results
            displayResults(result);
        } else {
            showError(result.error || 'An error occurred. Please try again.');
        }
    } catch (error) {
        showError('Network error. Please check your connection and try again.');
        console.error('Error:', error);
    } finally {
        hideLoader();
    }
});

// ==========================================
// Validation Functions
// ==========================================

function validateForm(data) {
    let isValid = true;

    // Age validation
    if (!data.age) {
        showFieldError('age', 'Age is required');
        isValid = false;
    } else if (isNaN(data.age) || data.age < 1 || data.age > 150) {
        showFieldError('age', 'Age must be between 1 and 150');
        isValid = false;
    }

    // Gender validation
    if (!data.gender) {
        showFieldError('gender', 'Gender is required');
        isValid = false;
    }

    // BMI validation
    if (!data.bmi) {
        showFieldError('bmi', 'BMI is required');
        isValid = false;
    } else if (isNaN(data.bmi) || data.bmi < 10 || data.bmi > 100) {
        showFieldError('bmi', 'BMI must be between 10 and 100');
        isValid = false;
    }

    // BP validation
    if (!data.bp_systolic) {
        showFieldError('bp_systolic', 'Systolic BP is required');
        isValid = false;
    } else if (isNaN(data.bp_systolic) || data.bp_systolic < 50 || data.bp_systolic > 250) {
        showFieldError('bp_systolic', 'Systolic BP must be between 50 and 250');
        isValid = false;
    }

    if (!data.bp_diastolic) {
        showFieldError('bp_diastolic', 'Diastolic BP is required');
        isValid = false;
    } else if (isNaN(data.bp_diastolic) || data.bp_diastolic < 30 || data.bp_diastolic > 150) {
        showFieldError('bp_diastolic', 'Diastolic BP must be between 30 and 150');
        isValid = false;
    }

    // Blood Sugar validation
    if (!data.blood_sugar) {
        showFieldError('blood_sugar', 'Blood Sugar is required');
        isValid = false;
    } else if (isNaN(data.blood_sugar) || data.blood_sugar < 40 || data.blood_sugar > 600) {
        showFieldError('blood_sugar', 'Blood Sugar must be between 40 and 600');
        isValid = false;
    }

    // Symptoms validation
    if (!data.symptoms || data.symptoms.trim() === '') {
        showFieldError('symptoms', 'Symptoms are required');
        isValid = false;
    }

    return isValid;
}

function showFieldError(fieldId, message) {
    const errorElement = document.getElementById(fieldId + 'Error');
    if (errorElement) {
        errorElement.textContent = message;
        errorElement.classList.add('show');
    }
}

function clearErrors() {
    const errorElements = document.querySelectorAll('.error-message');
    errorElements.forEach(el => {
        el.textContent = '';
        el.classList.remove('show');
    });
}

// ==========================================
// Result Display Functions
// ==========================================

function displayResults(data) {
    // Update health data display
    const healthData = data.health_data;
    document.getElementById('metricAge').textContent = Math.round(healthData.Age);
    document.getElementById('metricGender').textContent = healthData.Gender;
    document.getElementById('metricBMI').textContent = healthData.BMI.toFixed(1);
    document.getElementById('metricBP').textContent =
        Math.round(healthData.BP_Systolic) + '/' + Math.round(healthData.BP_Diastolic);
    document.getElementById('metricSugar').textContent = Math.round(healthData.BloodSugar) + ' mg/dL';
    document.getElementById('metricSymptoms').textContent = healthData.Symptoms;

    // Update risk card
    const riskLevel = data.risk_level;
    const riskCardEl = document.getElementById('riskCard');

    // Remove previous risk class
    riskCardEl.classList.remove('low-risk', 'medium-risk', 'high-risk');

    // Add new risk class and update icon
    if (riskLevel === 'Low Risk') {
        riskCardEl.classList.add('low-risk');
        document.getElementById('riskIcon').textContent = '✓';
    } else if (riskLevel === 'Medium Risk') {
        riskCardEl.classList.add('medium-risk');
        document.getElementById('riskIcon').textContent = '⚠️';
    } else {
        riskCardEl.classList.add('high-risk');
        document.getElementById('riskIcon').textContent = '🔴';
    }

    document.getElementById('riskLevel').textContent = riskLevel;
    document.getElementById('confidenceValue').textContent = data.confidence + '%';

    // Update probability bars
    animateBar('barLow', data.probabilities.low);
    animateBar('barMedium', data.probabilities.medium);
    animateBar('barHigh', data.probabilities.high);

    document.getElementById('valueLow').textContent = data.probabilities.low + '%';
    document.getElementById('valueMedium').textContent = data.probabilities.medium + '%';
    document.getElementById('valueHigh').textContent = data.probabilities.high + '%';

    // Display advice
    const adviceList = document.getElementById('adviceList');
    adviceList.innerHTML = '';
    data.advice.forEach(advice => {
        const adviceItem = document.createElement('div');
        adviceItem.className = 'advice-item';
        adviceItem.textContent = advice;
        adviceList.appendChild(adviceItem);
    });

    // Display recommendations
    const recommendationsList = document.getElementById('recommendationsList');
    recommendationsList.innerHTML = '';
    data.recommendations.forEach(rec => {
        const recItem = document.createElement('div');
        recItem.className = 'recommendation-item';

        if (rec.includes('URGENT') || rec.includes('immediately')) {
            recItem.classList.add('urgent');
        } else if (rec.includes('Schedule') || rec.includes('Monitor') || rec.includes('Increase') || rec.includes('Reduce')) {
            recItem.classList.add('warning');
        } else {
            recItem.classList.add('success');
        }

        recItem.textContent = rec;
        recommendationsList.appendChild(recItem);
    });

    // Show results container
    formContainer.style.display = 'none';
    resultContainer.style.display = 'block';

    // Scroll to results
    setTimeout(() => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }, 100);
}

function animateBar(barId, percentage) {
    const bar = document.getElementById(barId);
    setTimeout(() => {
        bar.style.width = percentage + '%';
    }, 100);
}

// ==========================================
// Utility Functions
// ==========================================

function showLoader() {
    loader.style.display = 'flex';
    submitBtn.disabled = true;
}

function hideLoader() {
    loader.style.display = 'none';
    submitBtn.disabled = false;
}

function showError(message) {
    alert('Error: ' + message);
}

function resetForm() {
    form.reset();
    clearErrors();
    formContainer.style.display = 'block';
    resultContainer.style.display = 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ==========================================
// Input validation on change
// ==========================================

document.getElementById('age').addEventListener('input', function () {
    if (this.value && (isNaN(this.value) || this.value < 1 || this.value > 150)) {
        showFieldError('age', 'Age must be between 1 and 150');
    } else {
        clearFieldError('age');
    }
});

document.getElementById('bmi').addEventListener('input', function () {
    if (this.value && (isNaN(this.value) || this.value < 10 || this.value > 100)) {
        showFieldError('bmi', 'BMI must be between 10 and 100');
    } else {
        clearFieldError('bmi');
    }
});

document.getElementById('bp_systolic').addEventListener('input', function () {
    if (this.value && (isNaN(this.value) || this.value < 50 || this.value > 250)) {
        showFieldError('bp_systolic', 'Systolic BP must be between 50 and 250');
    } else {
        clearFieldError('bp_systolic');
    }
});

document.getElementById('bp_diastolic').addEventListener('input', function () {
    if (this.value && (isNaN(this.value) || this.value < 30 || this.value > 150)) {
        showFieldError('bp_diastolic', 'Diastolic BP must be between 30 and 150');
    } else {
        clearFieldError('bp_diastolic');
    }
});

document.getElementById('blood_sugar').addEventListener('input', function () {
    if (this.value && (isNaN(this.value) || this.value < 40 || this.value > 600)) {
        showFieldError('blood_sugar', 'Blood Sugar must be between 40 and 600');
    } else {
        clearFieldError('blood_sugar');
    }
});

function clearFieldError(fieldId) {
    const errorElement = document.getElementById(fieldId + 'Error');
    if (errorElement) {
        errorElement.textContent = '';
        errorElement.classList.remove('show');
    }
}
