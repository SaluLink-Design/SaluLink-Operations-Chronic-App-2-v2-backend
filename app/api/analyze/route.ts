import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { clinical_note } = body;

    if (!clinical_note) {
      return NextResponse.json(
        { error: 'Clinical note is required' },
        { status: 400 }
      );
    }

    // Call Python backend
    const backendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
    const response = await fetch(`${backendUrl}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ clinical_note }),
    });

    if (!response.ok) {
      throw new Error('Python backend request failed');
    }

    const data = await response.json();
    
    return NextResponse.json(data);
  } catch (error) {
    console.error('Analysis error:', error);
    return NextResponse.json(
      { error: 'Failed to analyze clinical note. Make sure the Python backend is running.' },
      { status: 500 }
    );
  }
}

