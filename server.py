import pandas as pd
from fastapi import FastAPI, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Optional
import os
from main import process_and_route_messages

app = FastAPI()


@app.post("/intent_predictor_router/")
async def classify_msgs(request: Request, file: Optional[UploadFile] = None):
    try:
        # If a file is uploaded
        if file:
            if not file.filename.endswith('.csv'):
                raise HTTPException(status_code=400, detail="File must be a CSV.")

            df = pd.read_csv(file.file, encoding='ISO-8859-1')
            if "channel" not in df.columns or "message_content" not in df.columns:
                raise HTTPException(status_code=400, detail="CSV must contain 'channel' and 'message_content' columns.")

            msgs = list(zip(df["channel"], df["message_content"]))

        # If JSON body is provided instead of a file
        else:
            body = await request.json()
            messages: List[str] = body.get("message_content")
            channel: Optional[str] = body.get("channel", "unknown")  # Optional channel fallback

            if not messages or not isinstance(messages, list):
                raise HTTPException(status_code=400,
                                    detail="Request body must contain 'message_content' as a list of strings.")

            msgs = [(channel, msg) for msg in messages]
            df = pd.DataFrame(msgs, columns=["channel", "message_content"])

        # Classify the msgs
        labels, routing_info, processing_costs, chroma_ids = process_and_route_messages(msgs)

        # Append results
        df["target_label"] = labels
        df["routing_info"] = routing_info
        df["processing_cost"] = processing_costs
        df["chroma_vector_id"] = chroma_ids

        # Save to CSV and return file
        output_path = os.path.join("output", f"final_output.csv")
        df.to_csv(output_path, index=False)
        return FileResponse(output_path, media_type='text/csv')

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file:
            file.file.close()
            # # Clean up if the file was saved
            # if os.path.exists("output.csv"):
            #     os.remove("output.csv")

