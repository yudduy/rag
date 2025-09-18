import { put } from "@vercel/blob";
import { NextResponse } from "next/server";
import { z } from "zod";
import { randomUUID } from "crypto";
import path from "path";

export const runtime = "nodejs" as const;
import { auth } from "@/app/(auth)/auth";

const FileSchema = z.object({
  file: z
    .instanceof(File)
    .refine((file) => file.size <= 5 * 1024 * 1024, {
      message: "File size should be less than 5MB",
    })
    .refine(
      (file) =>
        ["image/jpeg", "image/png", "application/pdf"].includes(file.type),
      {
        message: "File type should be JPEG, PNG, or PDF",
      },
    ),
});

function sanitizeFilename(originalFilename: string): string {
  // Get the file extension safely
  const ext = path.extname(originalFilename).toLowerCase();
  const allowedExtensions = ['.jpg', '.jpeg', '.png', '.pdf'];
  
  // Use a safe extension or default to empty string
  const safeExtension = allowedExtensions.includes(ext) ? ext : '';
  
  // Generate a unique filename with UUID
  const uniqueId = randomUUID();
  
  return `${uniqueId}${safeExtension}`;
}

export async function POST(request: Request) {
  const session = await auth();

  if (!session) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  if (request.body === null) {
    return new Response("Request body is empty", { status: 400 });
  }

  try {
    const formData = await request.formData();
    const file = formData.get("file") as File;
    const accessPreference = formData.get("access") as string;

    if (!file) {
      return NextResponse.json({ error: "No file uploaded" }, { status: 400 });
    }

    const validatedFile = FileSchema.safeParse({ file });

    if (!validatedFile.success) {
      const errorMessage = validatedFile.error.errors
        .map((error) => error.message)
        .join(", ");

      return NextResponse.json({ error: errorMessage }, { status: 400 });
    }

    // Check if Vercel Blob is configured
    if (!process.env.BLOB_READ_WRITE_TOKEN) {
      return NextResponse.json({ 
        error: "File upload service not configured. Please configure BLOB_READ_WRITE_TOKEN environment variable." 
      }, { status: 503 });
    }

    const originalFilename = file.name;
    const sanitizedFilename = sanitizeFilename(originalFilename);
    const fileBuffer = await file.arrayBuffer();

    // Determine file access level
    const sensitiveTypes = [
      'application/pdf',
      'application/msword',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'text/plain',
      'text/markdown'
    ];
    
    let access: "public" | "private";
    
    if (accessPreference && (accessPreference === "public" || accessPreference === "private")) {
      // Honor explicit user preference
      access = accessPreference as "public" | "private";
    } else {
      // Default based on file type - sensitive types are private by default
      access = sensitiveTypes.includes(file.type) ? "private" : "public";
    }

    try {
      const data = await put(sanitizedFilename, fileBuffer, {
        access: "public" as const,
      });

      return NextResponse.json(data);
    } catch (error) {
      console.error("Vercel Blob upload error:", error);
      return NextResponse.json({ 
        error: "Upload failed. Please check your Vercel Blob configuration." 
      }, { status: 500 });
    }
  } catch (error) {
    console.error("File upload processing error:", error);
    return NextResponse.json(
      { error: "Failed to process request" },
      { status: 500 },
    );
  }
}
