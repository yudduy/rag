#!/usr/bin/env tsx

/**
 * CLI script to migrate from MemoryVectorStore to Pinecone
 * Usage: npx tsx scripts/migrate-to-pinecone.ts
 */

import { runMigration } from '../lib/migrate-to-pinecone';

// Run the migration
runMigration().catch(console.error);
