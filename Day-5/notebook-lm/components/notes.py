# =============================================================================
# components/notes.py — Right panel: saved notes as collapsible cards
# =============================================================================

import streamlit as st
from utils.helpers import list_notes, download_all_notes, delete_note


def render_notes_panel():
    """
    Render the notes panel on the right side.
    Shows saved notes as collapsible cards with delete + download options.
    """
    st.markdown("### 📝 Saved Notes")

    notes = list_notes()

    if not notes:
        st.markdown(
            """
            <div style='
                text-align: center;
                padding: 2rem 1rem;
                opacity: 0.5;
                border: 2px dashed rgba(128,128,128,0.3);
                border-radius: 12px;
                margin-top: 1rem;
            '>
                <div style='font-size: 2rem'>📋</div>
                <p style='margin: 0.5rem 0 0; font-size: 0.9rem'>
                    No notes yet.<br>
                    Click <b>💾 Save</b> on any response<br>
                    to save it as a note.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Download all notes button
    all_notes_md = download_all_notes()
    st.download_button(
        label        = "⬇️ Download All Notes",
        data         = all_notes_md,
        file_name    = "notebooklm_notes.md",
        mime         = "text/markdown",
        use_container_width = True,
        type         = "secondary",
    )

    st.caption(f"{len(notes)} note{'s' if len(notes) != 1 else ''} saved")
    st.divider()

    # Render each note as a collapsible card
    for note in notes:
        with st.expander(
            label    = f"📌 {note['title']}",
            expanded = False,
        ):
            st.caption(f"🕐 {note['modified']}")
            st.markdown(note["content"])

            col1, col2 = st.columns([1, 1])
            with col1:
                st.download_button(
                    label     = "⬇️ Download",
                    data      = note["content"],
                    file_name = note["filename"],
                    mime      = "text/markdown",
                    key       = f"dl_{note['filename']}",
                    use_container_width = True,
                )
            with col2:
                if st.button(
                    label = "🗑️ Delete",
                    key   = f"del_note_{note['filename']}",
                    use_container_width = True,
                ):
                    if delete_note(note["filename"]):
                        st.toast("Note deleted", icon="🗑️")
                        st.rerun()
