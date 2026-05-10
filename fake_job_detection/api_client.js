// TrueHire AI - API Client
// This handles all communication with the Flask backend

const API_BASE = 'http://localhost:5000/api';

// ==================== AUTHENTICATION ====================

async function registerUser(firstName, lastName, email, password, role) {
    try {
        const response = await fetch(`${API_BASE}/auth/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({
                first_name: firstName,
                last_name: lastName,
                email: email,
                password: password,
                role: role
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            console.log('Registration successful');
            window.location.href = '/dashboard.html';
            return data;
        } else {
            throw new Error(data.error || 'Registration failed');
        }
    } catch (error) {
        console.error('Register error:', error);
        alert(error.message);
    }
}

async function loginUser(email, password) {
    try {
        const response = await fetch(`${API_BASE}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ email, password })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            console.log('Login successful');
            window.location.href = '/dashboard.html';
            return data;
        } else {
            throw new Error(data.error || 'Login failed');
        }
    } catch (error) {
        console.error('Login error:', error);
        alert(error.message);
    }
}

async function logout() {
    try {
        await fetch(`${API_BASE}/auth/logout`, {
            method: 'POST',
            credentials: 'include'
        });
        window.location.href = '/landing.html';
    } catch (error) {
        console.error('Logout error:', error);
    }
}

async function getCurrentUser() {
    try {
        const response = await fetch(`${API_BASE}/auth/user`, {
            method: 'GET',
            credentials: 'include'
        });
        
        if (response.ok) {
            return await response.json();
        }
        return null;
    } catch (error) {
        console.error('Get user error:', error);
        return null;
    }
}

// ==================== JOB ANALYSIS ====================

async function analyzeJob(jobText, jobTitle = 'Unknown', company = 'Unknown') {
    try {
        const response = await fetch(`${API_BASE}/analyze/job`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({
                job_text: jobText,
                job_title: jobTitle,
                company: company
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            console.log('Analysis successful:', data);
            return data;
        } else {
            throw new Error(data.error || 'Analysis failed');
        }
    } catch (error) {
        console.error('Analysis error:', error);
        throw error;
    }
}

async function getAnalysisHistory() {
    try {
        const response = await fetch(`${API_BASE}/analysis/history`, {
            credentials: 'include'
        });
        
        if (response.ok) {
            return await response.json();
        } else {
            throw new Error('Failed to fetch history');
        }
    } catch (error) {
        console.error('History error:', error);
        return null;
    }
}

// ==================== DASHBOARD ====================

async function getDashboardStats() {
    try {
        const response = await fetch(`${API_BASE}/dashboard/stats`, {
            credentials: 'include'
        });
        
        if (response.ok) {
            return await response.json();
        } else {
            throw new Error('Failed to fetch stats');
        }
    } catch (error) {
        console.error('Stats error:', error);
        return null;
    }
}

// ==================== UTILITY FUNCTIONS ====================

function displayAnalysisResult(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const html = `
        <div style="text-align:center">
            <div style="font-size:2rem;font-weight:bold;color:${data.color};margin-bottom:1rem">
                ${data.score}%
            </div>
            <div style="font-size:1.2rem;margin-bottom:1rem">${data.verdict}</div>
            <div style="text-align:left;margin-top:1rem">
                <h4>Signals Found:</h4>
                ${displaySignals(data.signals)}
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

function displaySignals(signals) {
    let html = '';
    
    if (signals.high.length > 0) {
        html += '<h5 style="color:#FCA5A5">🔴 High Risk:</h5><ul>';
        signals.high.forEach(signal => {
            html += `<li>${signal}</li>`;
        });
        html += '</ul>';
    }
    
    if (signals.medium.length > 0) {
        html += '<h5 style="color:#FCD34D">🟡 Medium Risk:</h5><ul>';
        signals.medium.forEach(signal => {
            html += `<li>${signal}</li>`;
        });
        html += '</ul>';
    }
    
    if (signals.low.length > 0) {
        html += '<h5 style="color:#86EFAC">🟢 Low Risk:</h5><ul>';
        signals.low.forEach(signal => {
            html += `<li>${signal}</li>`;
        });
        html += '</ul>';
    }
    
    return html || '<p>No signals detected</p>';
}

// ==================== REDIRECT HELPERS ====================

function redirectIfNotLoggedIn() {
    getCurrentUser().then(user => {
        if (!user) {
            window.location.href = '/login.html';
        }
    });
}

function goToDashboard() {
    window.location.href = '/dashboard.html';
}

function goToDetector() {
    window.location.href = '/detector.html';
}

function goToLanding() {
    window.location.href = '/landing.html';
}
