/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {},
  images: {
    remotePatterns: [],
  },
  webpack: (config, { isServer }) => {
    // Ignore pdf-parse test files during build
    config.resolve.alias = {
      ...config.resolve.alias,
    };
    
    // Add ignore-loader for pdf-parse test files
    config.module.rules.push({
      test: /node_modules\/pdf-parse\/test\//,
      use: 'ignore-loader',
    });

    return config;
  },
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'Referrer-Policy',
            value: 'origin-when-cross-origin',
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block',
          },
        ],
      },
    ];
  },
};

export default nextConfig;
