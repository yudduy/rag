/**
 * Server-Sent Events endpoint for RAG demonstration real-time updates
 * Since Next.js doesn't natively support WebSockets, we use SSE for real-time communication
 */

import { NextRequest } from 'next/server';
import { auth } from '@/app/(auth)/auth';
import { ragDemoManager } from '@/lib/rag-demonstration-manager';

export async function GET(request: NextRequest) {
  const session = await auth();
  
  if (!session?.user?.id) {
    return new Response('Unauthorized', { status: 401 });
  }

  // Set up Server-Sent Events
  const encoder = new TextEncoder();
  
  const stream = new ReadableStream({
    start(controller) {
      // Send initial connection message
      const data = `data: ${JSON.stringify({ 
        type: 'connection', 
        message: 'Connected to RAG demonstration events',
        timestamp: Date.now()
      })}\n\n`;
      controller.enqueue(encoder.encode(data));

      // Subscribe to RAG events for this user
      const unsubscribe = ragDemoManager.subscribe(session.user.id, (event) => {
        try {
          const data = `data: ${JSON.stringify(event)}\n\n`;
          controller.enqueue(encoder.encode(data));
        } catch (error) {
          console.error('Error sending SSE event:', error);
        }
      });

      // Send heartbeat every 30 seconds to keep connection alive
      const heartbeatInterval = setInterval(() => {
        try {
          const heartbeat = `data: ${JSON.stringify({ 
            type: 'heartbeat', 
            timestamp: Date.now() 
          })}\n\n`;
          controller.enqueue(encoder.encode(heartbeat));
        } catch (error) {
          console.error('Error sending heartbeat:', error);
          clearInterval(heartbeatInterval);
        }
      }, 30000);

      // Clean up when client disconnects
      request.signal.addEventListener('abort', () => {
        console.log('RAG demonstration SSE client disconnected');
        clearInterval(heartbeatInterval);
        unsubscribe();
        controller.close();
      });
    }
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET',
      'Access-Control-Allow-Headers': 'Cache-Control'
    }
  });
}
