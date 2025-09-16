/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {},
  images: {
    remotePatterns: [],
  },
  webpack: (config, { isServer }) => {
    // Exclude pdf-parse test files from bundling
    config.module.rules.push({
      test: /node_modules\/pdf-parse\/test/,
      use: 'ignore-loader',
    });
    
    // Alternative approach: exclude test files entirely
    config.resolve.alias = {
      ...config.resolve.alias,
    };
    
    config.externals = config.externals || [];
    if (isServer) {
      config.externals.push({
        'pdf-parse/test': 'commonjs pdf-parse/test',
      });
    }
    
    return config;
  },
};

export default nextConfig;
