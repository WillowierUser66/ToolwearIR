# 1. Use an official Python base image
FROM python:3.13-alpine

# 2. Set a working directory inside the container
WORKDIR /app

# 3. Copy only requirements first (for better build caching)
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code
COPY . .

# 6. (optional but recommended) Set environment variables
# Prevent Python from writing .pyc files and enables unbuffered output (logs show up immediately)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 7. Command to run the app
# Adjust this line to whatever starts your app
CMD ["python", "IR_ToolWearapp.py"]
