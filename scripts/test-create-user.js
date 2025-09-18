#!/usr/bin/env node

// Script to test user creation directly
const { drizzle } = require('drizzle-orm/postgres-js');
const postgres = require('postgres');
const { eq } = require('drizzle-orm');
const { genSaltSync, hashSync } = require('bcrypt-ts');
const { user } = require('../db/schema');

async function testCreateUser() {
  try {
    const connectionString = process.env.POSTGRES_URL;
    if (!connectionString) {
      console.error('POSTGRES_URL environment variable not set');
      process.exit(1);
    }

    const url = new URL(connectionString);
    url.searchParams.set('sslmode', 'require');
    
    const client = postgres(url.toString(), { max: 1 });
    const db = drizzle(client);

    console.log('Testing user creation...');
    
    const testEmail = 'test-user@example.com';
    const testPassword = 'testpassword123';

    // Hash password
    const salt = genSaltSync(10);
    const hashedPassword = hashSync(testPassword, salt);

    console.log('Attempting to create user:', testEmail);
    
    try {
      const [newUser] = await db
        .insert(user)
        .values({ 
          email: testEmail, 
          password: hashedPassword 
        })
        .returning();
      
      console.log('✅ User created successfully!');
      console.log('User ID:', newUser.id);
      console.log('User Email:', newUser.email);
      console.log('Has Password:', !!newUser.password);
      
      // Clean up - delete the test user
      await db.delete(user).where(eq(user.id, newUser.id));
      console.log('✅ Test user cleaned up');
      
    } catch (createError) {
      console.error('❌ Failed to create user:', createError.message);
      console.error('Full error:', createError);
    }

    await client.end();
  } catch (error) {
    console.error('❌ Script error:', error);
    process.exit(1);
  }
}

testCreateUser();
