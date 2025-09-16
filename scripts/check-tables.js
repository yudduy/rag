#!/usr/bin/env node

// Script to check database tables and structure
const postgres = require('postgres');

async function checkTables() {
  try {
    // Database connection
    const connectionString = process.env.POSTGRES_URL;
    if (!connectionString) {
      console.error('POSTGRES_URL environment variable not set');
      process.exit(1);
    }

    const client = postgres(connectionString, { max: 1 });

    console.log('Checking database tables...');
    
    // Check if tables exist
    const tables = await client`
      SELECT table_name, table_schema 
      FROM information_schema.tables 
      WHERE table_schema = 'public' 
      ORDER BY table_name;
    `;
    
    console.log('Tables in database:', tables.length);
    tables.forEach(table => {
      console.log(`- ${table.table_name} (schema: ${table.table_schema})`);
    });

    // Check User table structure specifically
    if (tables.some(t => t.table_name === 'User')) {
      console.log('\nUser table columns:');
      const columns = await client`
        SELECT column_name, data_type, character_maximum_length, is_nullable, column_default
        FROM information_schema.columns 
        WHERE table_name = 'User' AND table_schema = 'public'
        ORDER BY ordinal_position;
      `;
      
      columns.forEach(col => {
        console.log(`- ${col.column_name}: ${col.data_type}${col.character_maximum_length ? `(${col.character_maximum_length})` : ''} ${col.is_nullable === 'YES' ? 'NULL' : 'NOT NULL'} ${col.column_default ? `DEFAULT ${col.column_default}` : ''}`);
      });
    }

    // Check for any existing users
    try {
      const userCount = await client`SELECT COUNT(*) as count FROM "User"`;
      console.log(`\nTotal users in database: ${userCount[0].count}`);
    } catch (error) {
      console.log('\nCould not query User table:', error.message);
    }

    await client.end();
  } catch (error) {
    console.error('Error checking tables:', error);
    process.exit(1);
  }
}

checkTables();
