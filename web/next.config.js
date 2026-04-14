/** @type {import('next').NextConfig} */
const nextConfig = {
  // Needed for the standalone Docker image
  output: "standalone",

  async rewrites() {
    const apiBase = process.env.API_URL || "http://localhost:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${apiBase}/api/:path*`,
      },
      {
        source: "/ws/:path*",
        destination: `${apiBase}/ws/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
