import streamlit as st

def setup_chat_page_header():
    st.set_page_config(page_title="Northwind HR Assistant", page_icon="🏢")
    st.html("""
        <style>
        /* only messages that have the default user avatar */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            flex-direction: row-reverse;   /* avatar goes to the right */
            width: fit-content;            /* bubble hugs the text */
            max-width: 80%;
            margin-left: auto;             /* push bubble to the right */
        }
        </style>
    """)
    
    st.title(
        "Welcome to :primary[Northwind Technologies Inc.]",
        text_alignment="center",
        anchor=False,  # hides the link icon on hover
    )
    st.caption(
        "Ask me anything about company policies, leave, benefits and conduct.",
        text_alignment="center",
    )
    st.divider()