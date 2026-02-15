// shared utility functions

const Utils = {
    // ---------------------------------------------------------------
    // Auth token management
    // ---------------------------------------------------------------
    getToken() {
        // Priority: localStorage > URL param
        const stored = localStorage.getItem('api_monitor_token');
        if (stored) return stored;
        const params = new URLSearchParams(window.location.search);
        const fromUrl = params.get('token');
        if (fromUrl) {
            localStorage.setItem('api_monitor_token', fromUrl);
            return fromUrl;
        }
        return '';
    },

    setToken(token) {
        if (token) {
            localStorage.setItem('api_monitor_token', token);
        } else {
            localStorage.removeItem('api_monitor_token');
        }
    },

    authHeaders() {
        const token = this.getToken();
        if (!token) return {};
        return { 'Authorization': `Bearer ${token}` };
    },

    /**
     * Wrapper around fetch that includes auth headers.
     * On 401, prompts user for token.
     */
    async authFetch(url, options = {}) {
        const headers = { ...this.authHeaders(), ...(options.headers || {}) };
        const res = await fetch(url, { ...options, headers });
        if (res.status === 401) {
            const token = prompt('Please enter API_MONITOR_TOKEN:');
            if (token) {
                this.setToken(token);
                // Retry with new token
                const retryHeaders = { ...this.authHeaders(), ...(options.headers || {}) };
                return fetch(url, { ...options, headers: retryHeaders });
            }
        }
        return res;
    },

    /**
     * Create an SSE EventSource with token support.
     */
    createEventSource(url) {
        const token = this.getToken();
        if (token) {
            const sep = url.includes('?') ? '&' : '?';
            return new EventSource(`${url}${sep}token=${encodeURIComponent(token)}`);
        }
        return new EventSource(url);
    },

    // ---------------------------------------------------------------
    // Format timestamp (seconds) to readable string
    // ---------------------------------------------------------------
    fmtTime(ts) {
        if (!ts) return '-';
        const d = new Date(ts * 1000);
        const now = new Date();
        if (d.toDateString() === now.toDateString()) {
            return d.toLocaleTimeString('en-US', { hour12: false });
        }
        return d.toLocaleDateString('en-US', { month: '2-digit', day: '2-digit' }) + ' ' + d.toLocaleTimeString('en-US', { hour12: false });
    },

    fmtFullTime(ts) {
        if (!ts) return '-';
        return new Date(ts * 1000).toLocaleString('en-US', { hour12: false });
    },

    // Format duration (seconds) to ms string + color class
    fmtDuration(sec) {
        const ms = Math.round(sec * 1000);
        let colorClass = 'text-emerald-500';
        if (ms > 1000) colorClass = 'text-amber-500';
        if (ms > 3000) colorClass = 'text-rose-500';
        return { text: `${ms}ms`, class: colorClass, ms };
    },

    // Get status color (Tailwind classes)
    getStatusColor(status, running) {
        if (running) return { dot: 'bg-sky-500 animate-pulse', text: 'text-sky-500', bg: 'bg-sky-500/10' };
        if (status === 'healthy') return { dot: 'bg-emerald-500', text: 'text-emerald-500', bg: 'bg-emerald-500/10' };
        if (status === 'degraded') return { dot: 'bg-amber-500', text: 'text-amber-500', bg: 'bg-amber-500/10' };
        if (['down', 'error'].includes(status)) return { dot: 'bg-rose-500', text: 'text-rose-500', bg: 'bg-rose-500/10' };
        return { dot: 'bg-zinc-500', text: 'text-zinc-500', bg: 'bg-zinc-500/10' };
    },

    // Theme Management
    initTheme() {
        if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
    },

    toggleTheme() {
        if (document.documentElement.classList.contains('dark')) {
            document.documentElement.classList.remove('dark');
            localStorage.theme = 'light';
        } else {
            document.documentElement.classList.add('dark');
            localStorage.theme = 'dark';
        }
    }
};

// Expose to window
window.Utils = Utils;

// Init theme immediately
Utils.initTheme();
