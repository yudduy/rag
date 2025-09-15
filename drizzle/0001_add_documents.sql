-- Migration: Add documents table
-- Created: 2024

CREATE TABLE IF NOT EXISTS "Document" (
  "id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
  "filename" varchar(255) NOT NULL,
  "originalName" varchar(255) NOT NULL,
  "fileType" varchar(50) NOT NULL,
  "fileSize" integer NOT NULL,
  "content" text NOT NULL,
  "status" varchar(20) DEFAULT 'indexed' NOT NULL,
  "chunkCount" integer DEFAULT 0 NOT NULL,
  "createdAt" timestamp DEFAULT now() NOT NULL,
  "updatedAt" timestamp DEFAULT now() NOT NULL,
  "userId" uuid NOT NULL
);

-- Add foreign key constraint
DO $$ BEGIN
  ALTER TABLE "Document" ADD CONSTRAINT "Document_userId_User_id_fk" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE cascade;
EXCEPTION
  WHEN duplicate_object THEN null;
END $$;

-- Add indexes for better performance
CREATE INDEX IF NOT EXISTS "idx_document_user_id" ON "Document"("userId");
CREATE INDEX IF NOT EXISTS "idx_document_status" ON "Document"("status");
CREATE INDEX IF NOT EXISTS "idx_document_created_at" ON "Document"("createdAt" DESC);
