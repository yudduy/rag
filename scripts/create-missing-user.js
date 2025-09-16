#!/usr/bin/env node

// Script to create the missing user from the session
const { drizzle } = require('drizzle-orm/postgres-js');
const postgres = require('postgres');
const { pgTable, varchar, uuid, unique } = require('drizzle-orm/pg-core');
const { genSaltSync, hashSync } = require('bcrypt-ts');

// Define user schema
const user = pgTable("User", {
  id: uuid("id").primaryKey().notNull().defaultRandom(),
  email: varchar("email", { length: 64 }).notNull(),
  password: varchar("password", { length: 128 }),
}, (table) => ({
  uniqueEmail: unique().on(table.email),
}));

async function createMissingUser() {
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

    // Create the missing user from the session logs
    const sessionEmail = 'kduynguy@gmail.com';
    const sessionUserId = 'b1146900-7284-4e3a-910e-b730847f4db8';
    
    // Use a default password (user will need to reset it)
    const defaultPassword = 'temp123456';
    const salt = genSaltSync(10);
    const hashedPassword = hashSync(defaultPassword, salt);

    console.log('Creating missing user from session...');
    console.log('Email:', sessionEmail);
    console.log('Expected ID:', sessionUserId);
    
    try {
      // Try to create user with the specific ID from the session
      const result = await client`
        INSERT INTO "User" (id, email, password) 
        VALUES (${sessionUserId}, ${sessionEmail}, ${hashedPassword})
        RETURNING id, email;
      `;
      
      console.log('✅ User created successfully!');
      console.log('User ID:', result[0].id);
      console.log('User Email:', result[0].email);
      console.log('⚠️  Default password set to: temp123456');
      console.log('🔒 User should change password on next login');
      
    } catch (createError) {
      console.error('❌ Failed to create user:', createError.message);
      
      // If the specific UUID fails, try creating without specifying ID
      console.log('Trying to create user with auto-generated ID...');
      try {
        const [newUser] = await db
          .insert(user)
          .values({ 
            email: sessionEmail, 
            password: hashedPassword 
          })
          .returning();
        
        console.log('✅ User created with new ID!');
        console.log('New User ID:', newUser.id);
        console.log('⚠️  Session will need to be refreshed to use new ID');
        
      } catch (fallbackError) {
        console.error('❌ Fallback creation also failed:', fallbackError.message);
      }
    }

    await client.end();
  } catch (error) {
    console.error('❌ Script error:', error);
    process.exit(1);
  }
}

createMissingUser();
