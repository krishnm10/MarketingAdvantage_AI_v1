/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    API_BASE_URL: "http://localhost:8000/api/v2", // FastAPI backend
  },
  webpack(config) {
    // optional: helps with module resolution for cleaner imports
    config.resolve.alias["@"] = __dirname;
    return config;
  },
};

module.exports = nextConfig;
