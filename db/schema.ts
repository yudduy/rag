import { Message } from "ai";
import { InferSelectModel } from "drizzle-orm";
import {
  pgTable,
  varchar,
  timestamp,
  json,
  uuid,
  boolean,
  text,
  integer,
  unique,
} from "drizzle-orm/pg-core";
import { sql } from "drizzle-orm";

export const user = pgTable("User", {
  id: uuid("id").primaryKey().notNull().defaultRandom(),
  email: varchar("email", { length: 64 }).notNull(),
  password: varchar("password", { length: 128 }),
}, (table) => ({
  uniqueEmail: unique().on(table.email),
}));

export type User = InferSelectModel<typeof user>;

export const chat = pgTable("Chat", {
  id: uuid("id").primaryKey().notNull().defaultRandom(),
  createdAt: timestamp("createdAt").notNull(),
  messages: json("messages").notNull(),
  userId: uuid("userId")
    .notNull()
    .references(() => user.id),
});

export type Chat = Omit<InferSelectModel<typeof chat>, "messages"> & {
  messages: Array<Message>;
};

export const reservation = pgTable("Reservation", {
  id: uuid("id").primaryKey().notNull().defaultRandom(),
  createdAt: timestamp("createdAt").notNull(),
  details: json("details").notNull(),
  hasCompletedPayment: boolean("hasCompletedPayment").notNull().default(false),
  userId: uuid("userId")
    .notNull()
    .references(() => user.id),
});

export type Reservation = InferSelectModel<typeof reservation>;

export const document = pgTable("Document", {
  id: uuid("id").primaryKey().notNull().defaultRandom(),
  filename: varchar("filename", { length: 255 }).notNull(),
  originalName: varchar("originalName", { length: 255 }).notNull(),
  fileType: varchar("fileType", { length: 50 }).notNull(),
  fileSize: integer("fileSize").notNull(),
  content: text("content").notNull(),
  status: varchar("status", { length: 20 }).notNull().default("indexed"), // pending, indexing, indexed, failed
  chunkCount: integer("chunkCount").notNull().default(0),
  createdAt: timestamp("createdAt").notNull().default(sql`CURRENT_TIMESTAMP`),
  updatedAt: timestamp("updatedAt").notNull().default(sql`CURRENT_TIMESTAMP`),
  userId: uuid("userId")
    .notNull()
    .references(() => user.id, { onDelete: "cascade" }),
});

export type Document = InferSelectModel<typeof document>;
