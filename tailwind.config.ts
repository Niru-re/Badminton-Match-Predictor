/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        court: {
          DEFAULT: "#1a472a",
          light: "#2d6a4f",
          dark: "#0d2818",
        },
        shuttle: {
          DEFAULT: "#f4d03f",
          dim: "#d4ac0d",
        },
      },
      fontFamily: {
        sans: ["var(--font-geist-sans)", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
