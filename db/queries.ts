import "server-only";

import { genSaltSync, hashSync } from "bcrypt-ts";
import { desc, eq } from "drizzle-orm";
import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";

import { user, chat, User, reservation, document, Document } from "./schema";

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
  max: 1,
  // Prevent connections during build
  transform: process.env.NODE_ENV === 'production' && !process.env.POSTGRES_URL 
    ? undefined 
    : postgres.camel
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
  messages: any;
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
        .where(eq(chat.id, id));
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

export async function deleteChatById({ id }: { id: string }) {
  try {
    return await db.delete(chat).where(eq(chat.id, id));
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

export async function getChatById({ id }: { id: string }) {
  try {
    const [selectedChat] = await db.select().from(chat).where(eq(chat.id, id));
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
  return await db.insert(reservation).values({
    id,
    createdAt: new Date(),
    userId,
    hasCompletedPayment: false,
    details: JSON.stringify(details),
  });
}

export async function getReservationById({ id }: { id: string }) {
  const [selectedReservation] = await db
    .select()
    .from(reservation)
    .where(eq(reservation.id, id));

  return selectedReservation;
}

export async function updateReservation({
  id,
  hasCompletedPayment,
}: {
  id: string;
  hasCompletedPayment: boolean;
}) {
  return await db
    .update(reservation)
    .set({
      hasCompletedPayment,
    })
    .where(eq(reservation.id, id));
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
    console.error("Failed to get documents by user from database");
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
