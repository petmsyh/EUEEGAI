from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import os
from dotenv import load_dotenv
import os

load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")

print(f"Using API URL: {BOT_TOKEN}")
# Replace with your bot token from BotFather

# Command to start the bot
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Welcome to my Streamlit Telegram Bot! Send a message to interact.")

# Handle text messages
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    # Example: Echo the user's message (you can integrate with Streamlit here)
    response = f"You said: {user_message}"
    await update.message.reply_text(response)

def main():
    # Initialize the bot
    application = Application.builder().token(BOT_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start polling (for local testing)
    print("Telegram Bot started!")
    application.run_polling()

if __name__ == "__main__":
    main()
