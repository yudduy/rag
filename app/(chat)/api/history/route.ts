import { auth } from "@/app/(auth)/auth";
import { getChatsByUserId } from "@/db/queries";

export async function GET() {
  const session = await auth();

  if (!session || !session.user || !session.user.id) {
    return Response.json("Unauthorized!", { status: 401 });
  }

  try {
    const chats = await getChatsByUserId({ id: session.user.id });
    return Response.json(chats);
  } catch (error) {
    console.error("Failed to load chats:", error);
    return Response.json({ error: 'Failed to load chats' }, { status: 500 });
  }
}
