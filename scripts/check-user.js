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

    // Check for the specific user from the logs
    const testEmail = 'kduynguy@gmail.com';
    const testUserId = 'b1146900-7284-4e3a-910e-b730847f4db8';

    console.log('Checking database for user...');
    console.log('Email:', testEmail);
    console.log('Session User ID:', testUserId);
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
