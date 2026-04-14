/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        zeno: {
          bg:      "#0a0e1a",
          card:    "#111827",
          border:  "#1f2937",
          accent:  "#6366f1",
          green:   "#10b981",
          red:     "#ef4444",
          yellow:  "#f59e0b",
          muted:   "#6b7280",
        },
      },
    },
  },
  plugins: [],
};
