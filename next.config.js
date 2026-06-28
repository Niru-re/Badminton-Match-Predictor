/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    // In local dev, proxy /api/* to the Python Flask server on port 5001.
    // On Vercel, the Python serverless functions handle /api/* directly.
    if (process.env.NODE_ENV === "development") {
      return [
        {
          source: "/api/:path*",
          destination: "http://localhost:5001/api/:path*",
        },
      ];
    }
    return [];
  },
};

module.exports = nextConfig;
