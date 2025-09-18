import "server-only";

import { genSaltSync, hashSync } from "bcrypt-ts";
import { desc, eq, and } from "drizzle-orm";
import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";

import { user, chat, User, reservation, document, Document } from "./schema";
import { Message } from "ai";

// Optionally, if not using email/pass login, you can
// use the Drizzle adapter for Auth.js / NextAuth
// https://authjs.dev/reference/adapter/drizzle

// Handle build-time when POSTGRES_URL might not be available
const connectionString = process.env.POSTGRES_URL 
  ? (() => {
      const url = new URL(process.env.POSTGRES_URL);
      url.searchParams.set('sslmode', 'require');
      return url.toString();
    })()
  : 'postgres://dummy:dummy@localhost:5432/dummy';

let client = postgres(connectionString, { 
  max: parseInt(process.env.DB_MAX_POOL || '10', 10),
  // Configurable transform option
  transform: process.env.POSTGRES_USE_CAMEL === 'true' ? postgres.camel : undefined
});
let db = drizzle(client);

export async function getUser(email: string): Promise<Array<User>> {
  try {
    return await db.select().from(user).where(eq(user.email, email));
  } catch (error) {
    console.error("Failed to get user from database");
    throw error;
  }
}

export async function createUser(email: string, password: string): Promise<User> {
  let salt = genSaltSync(10);
  let hash = hashSync(password, salt);

  try {
    const [newUser] = await db
      .insert(user)
      .values({ email, password: hash })
      .returning();
    
    return newUser;
  } catch (error) {
    console.error("Failed to create user in database");
    throw error;
  }
}

export async function saveChat({
  id,
  messages,
  userId,
}: {
  id: string;
  messages: Array<Message>;
  userId: string;
}) {
  try {
    const selectedChats = await db.select().from(chat).where(eq(chat.id, id));

    if (selectedChats.length > 0) {
      return await db
        .update(chat)
        .set({
          messages: JSON.stringify(messages),
        })
        .where(and(eq(chat.id, id), eq(chat.userId, userId)));
    }

    return await db.insert(chat).values({
      id,
      createdAt: new Date(),
      messages: JSON.stringify(messages),
      userId,
    });
  } catch (error) {
    console.error("Failed to save chat in database");
    throw error;
  }
}

export async function deleteChatById({ id, userId }: { id: string; userId: string }) {
  try {
    return await db.delete(chat).where(and(eq(chat.id, id), eq(chat.userId, userId)));
  } catch (error) {
    console.error("Failed to delete chat by id from database");
    throw error;
  }
}

export async function getChatsByUserId({ id }: { id: string }) {
  try {
    return await db
      .select()
      .from(chat)
      .where(eq(chat.userId, id))
      .orderBy(desc(chat.createdAt));
  } catch (error) {
    console.error("Failed to get chats by user from database");
    throw error;
  }
}

export async function getChatById({ id, userId }: { id: string; userId: string }) {
  try {
    const [selectedChat] = await db
      .select()
      .from(chat)
      .where(and(eq(chat.id, id), eq(chat.userId, userId)));
    return selectedChat;
  } catch (error) {
    console.error("Failed to get chat by id from database");
    throw error;
  }
}

export async function createReservation({
  id,
  userId,
  details,
}: {
  id: string;
  userId: string;
  details: any;
}) {
  try {
    return await db.insert(reservation).values({
      id,
      createdAt: new Date(),
      userId,
      hasCompletedPayment: false,
      details: JSON.stringify(details),
    });
  } catch (error) {
    console.error(`Failed to create reservation for user ${userId} with id ${id}:`, error);
    throw error;
  }
}

export async function getReservationById({ id }: { id: string }) {
  try {
    const [selectedReservation] = await db
      .select()
      .from(reservation)
      .where(eq(reservation.id, id));

    return selectedReservation;
  } catch (error) {
    console.error(`Failed to get reservation by id ${id}:`, error);
    throw error;
  }
}

export async function updateReservation({
  id,
  hasCompletedPayment,
}: {
  id: string;
  hasCompletedPayment: boolean;
}) {
  try {
    return await db
      .update(reservation)
      .set({
        hasCompletedPayment,
      })
      .where(eq(reservation.id, id));
  } catch (error) {
    console.error(`Failed to update reservation ${id}:`, error);
    throw error;
  }
}

// Document queries
export async function createDocument({
  filename,
  originalName,
  fileType,
  fileSize,
  content,
  userId,
}: {
  filename: string;
  originalName: string;
  fileType: string;
  fileSize: number;
  content: string;
  userId: string;
}): Promise<Document> {
  try {
    const [newDocument] = await db
      .insert(document)
      .values({
        filename,
        originalName,
        fileType,
        fileSize,
        content,
        userId,
        status: "indexed",
        chunkCount: 0,
      })
      .returning();
    
    return newDocument;
  } catch (error) {
    console.error("Failed to create document in database");
    throw error;
  }
}

export async function getDocumentsByUserId({ userId }: { userId: string }): Promise<Document[]> {
  try {
    return await db
      .select()
      .from(document)
      .where(eq(document.userId, userId))
      .orderBy(desc(document.createdAt));
  } catch (error) {
    console.error("Failed to get documents by user from database:", error);
    throw error;
  }
}

export async function getDocumentById({ id }: { id: string }): Promise<Document | undefined> {
  try {
    const [selectedDocument] = await db
      .select()
      .from(document)
      .where(eq(document.id, id));
    return selectedDocument;
  } catch (error) {
    console.error("Failed to get document by id from database");
    throw error;
  }
}

export async function updateDocumentChunkCount({
  id,
  chunkCount,
}: {
  id: string;
  chunkCount: number;
}) {
  try {
    return await db
      .update(document)
      .set({
        chunkCount,
        updatedAt: new Date(),
      })
      .where(eq(document.id, id));
  } catch (error) {
    console.error("Failed to update document chunk count");
    throw error;
  }
}

export async function deleteDocumentById({ id }: { id: string }) {
  try {
    return await db.delete(document).where(eq(document.id, id));
  } catch (error) {
    console.error("Failed to delete document by id from database");
    throw error;
  }
}
