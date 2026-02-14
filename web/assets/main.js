// shared utility functions

const Utils = {
    // Format timestamp (seconds) to readable string
    fmtTime(ts) {
        if (!ts) return '-';
        const d = new Date(ts * 1000);
        // Check if today
        const now = new Date();
        if (d.toDateString() === now.toDateString()) {
            return d.toLocaleTimeString('zh-CN', { hour12: false });
        }
        return d.toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' }) + ' ' + d.toLocaleTimeString('zh-CN', { hour12: false });
    },

    fmtFullTime(ts) {
        if (!ts) return '-';
        return new Date(ts * 1000).toLocaleString('zh-CN', { hour12: false });
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
