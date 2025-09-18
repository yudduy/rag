import { compare } from "bcrypt-ts";
import NextAuth, { User, Session } from "next-auth";
import { JWT } from "next-auth/jwt";
import Credentials from "next-auth/providers/credentials";

import { getUser } from "@/db/queries";

import { authConfig } from "./auth.config";

interface ExtendedSession extends Session {
  user: User;
}

interface Credentials {
  email: string;
  password: string;
}

export const {
  handlers: { GET, POST },
  auth,
  signIn,
  signOut,
} = NextAuth({
  ...authConfig,
  providers: [
    Credentials({
      credentials: {},
      async authorize(credentials: any) {
        const { email, password } = credentials as Credentials;
        if (!email || !password) return null;
        
        let users = await getUser(email);
        if (users.length === 0) return null;
        
        const user = users[0];
        if (!user.password) return null;
        
        const passwordsMatch = await compare(password, user.password);
        if (passwordsMatch) {
          const safeUser: User = {
            id: user.id,
            email: user.email,
            name: (user as any).name ?? undefined,
            image: (user as any).image ?? undefined,
          };
          return safeUser;
        }
        
        return null;
      },
    }),
  ],
  callbacks: {
    async jwt({ token, user }: { token: JWT & { id?: string }; user?: User }) {
      if (user) {
        token.id = user.id;
      }

      return token;
    },
    async session({
      session,
      token,
    }: {
      session: ExtendedSession;
      token: JWT & { id?: string };
    }) {
      if (session.user) {
        session.user.id = token.id as string;
      }

      return session;
    },
  },
});
