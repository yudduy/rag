import { NextAuthConfig } from "next-auth";

export const authConfig = {
  pages: {
    signIn: "/login",
    newUser: "/",
  },
  providers: [
    // added later in auth.ts since it requires bcrypt which is only compatible with Node.js
    // while this file is also used in non-Node.js environments
  ],
  callbacks: {
    authorized({ auth, request: { nextUrl } }) {
      let isLoggedIn = !!auth?.user;
      let isOnChat = nextUrl.pathname === "/" || nextUrl.pathname.startsWith("/chat");
      let isOnRegister = nextUrl.pathname.startsWith("/register");
      let isOnLogin = nextUrl.pathname.startsWith("/login");

      if (isOnLogin || isOnRegister) {
        return isLoggedIn ? Response.redirect(new URL("/", nextUrl)) : true;
      }
      if (isOnChat) {
        return isLoggedIn ? true : Response.redirect(new URL("/login", nextUrl));
      }
      // All other routes are public by default
      return true;
    },
  },
  secret: process.env.AUTH_SECRET,
} satisfies NextAuthConfig;
