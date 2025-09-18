#!/usr/bin/env node

// Script to check user existence in database
const { drizzle } = require('drizzle-orm/postgres-js');
const postgres = require('postgres');
const { eq } = require('drizzle-orm');
const { pgTable, varchar, uuid, unique } = require('drizzle-orm/pg-core');

// Define user schema directly (since we can't import TS modules in Node.js easily)
const user = pgTable("User", {
  id: uuid("id").primaryKey().notNull().defaultRandom(),
  email: varchar("email", { length: 64 }).notNull(),
  password: varchar("password", { length: 128 }),
}, (table) => ({
  uniqueEmail: unique().on(table.email),
}));

async function checkUser() {
  try {
    // Database connection
    const connectionString = process.env.POSTGRES_URL;
    if (!connectionString) {
      console.error('POSTGRES_URL environment variable not set');
      process.exit(1);
    }

    const url = new URL(connectionString);
    url.searchParams.set('sslmode', 'require');
    
    const client = postgres(url.toString(), { max: 1 });
    const db = drizzle(client);

    // Get user details from environment variables or command line arguments
    const testEmail = process.env.CHECK_USER_EMAIL || process.argv[2];
    const testUserId = process.env.CHECK_USER_ID || process.argv[3];

    if (!testEmail || !testUserId) {
      console.error('Error: Missing required parameters');
      console.error('Usage: node check-user.js <email> <user-id>');
      console.error('Or set CHECK_USER_EMAIL and CHECK_USER_ID environment variables');
      process.exit(1);
    }

    console.log('Checking database for user...');
    console.log('Email:', testEmail.replace(/(.{2}).*@/, '$1***@')); // Mask email
    console.log('Session User ID:', testUserId.slice(-8)); // Show only last 8 chars
    console.log('---');

    // Query by email
    const usersByEmail = await db.select().from(user).where(eq(user.email, testEmail));
    console.log('Users found by email:', usersByEmail.length);
    if (usersByEmail.length > 0) {
      usersByEmail.forEach((u, idx) => {
        console.log(`User ${idx + 1}:`, {
          id: u.id,
          email: u.email,
          hasPassword: !!u.password
        });
      });
    }

    // Query by ID
    const usersById = await db.select().from(user).where(eq(user.id, testUserId));
    console.log('\nUsers found by session ID:', usersById.length);
    if (usersById.length > 0) {
      usersById.forEach((u, idx) => {
        console.log(`User ${idx + 1}:`, {
          id: u.id,
          email: u.email,
          hasPassword: !!u.password
        });
      });
    }

    // Get all users (limited to 10 for safety)
    const allUsers = await db.select().from(user).limit(10);
    console.log('\nAll users in database (max 10):', allUsers.length);
    allUsers.forEach((u, idx) => {
      console.log(`User ${idx + 1}:`, {
        id: u.id,
        email: u.email,
        hasPassword: !!u.password
      });
    });

    await client.end();
  } catch (error) {
    console.error('Error checking user:', error);
    process.exit(1);
  }
}

checkUser();
