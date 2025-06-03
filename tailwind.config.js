/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        amberOklch: "oklch(0.656 0.241 354.308)",
        peach: "#eb8951",
      },
    },
  },
  plugins: [],
}

