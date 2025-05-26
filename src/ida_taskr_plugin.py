import configparser
import json
import os
import pathlib
from datetime import datetime
from textwrap import dedent

# For IDA integration
import idaapi
from PyQt5.QtCore import QEvent, Qt, QTimer
from PyQt5.QtGui import (
    QColor,
    QFont,
    QIcon,
    QKeySequence,
    QPalette,
    QPixmap,
    QTextCursor,
)
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QSplitter,
    QStyleFactory,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Configuration paths
USER_IDADIR = pathlib.Path(idaapi.get_user_idadir())
USER_CFGDIR = USER_IDADIR / "cfg"
QTAPP_STYLE_CFG = USER_CFGDIR / "qtappstyle.cfg"
PLUGIN_NAME = "IDAThemr"


def load_style():
    """Read the previously selected style and dark mode setting from the config file."""
    config = configparser.ConfigParser()
    try:
        config.read(QTAPP_STYLE_CFG)
    except Exception:
        pass

    style = config.get("default", "qt_style", fallback="Fusion")
    dark_mode = config.getboolean("default", "dark_mode", fallback=False)
    return style, dark_mode


def save_style(style: str, dark_mode: bool = False):
    """Save the chosen style and dark mode setting to the config file."""
    config = configparser.ConfigParser()
    config["default"] = {
        "qt_style": style,
        "dark_mode": str(dark_mode).lower(),
    }
    USER_CFGDIR.mkdir(parents=True, exist_ok=True)
    with open(QTAPP_STYLE_CFG, "w") as configfile:
        config.write(configfile)


def toggle_dark_theme(on: bool, property_name: str = "os-dark-theme"):
    """Toggle the OS theme for the IDA GUI."""
    app = QApplication.instance()
    # Flip the dynamic property on every widget
    for w in app.allWidgets():
        w.setProperty(property_name, on)
        w.style().unpolish(w)
        w.style().polish(w)
        w.update()


class StyleSheetWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Force widget font across the board, and monospace in the editor
        self.setStyleSheet(
            """
            QWidget {
                font-family: -apple-system, system-ui, BlinkMacSystemFont,
                             "Segoe UI", Roboto, "Helvetica Neue",
                             Arial, sans-serif !important;
            }
            QTextEdit {
                font-family: "Maple Mono NF Light", "Cascadia Code", "Consolas" !important;
                font-size: 10em !important;
            }
        """
        )

        self.tape = []
        self.tape_pos = -1
        self.current_file = None

        # Create main layout
        layout = QVBoxLayout(self)

        # Add file path display
        self.file_path_label = QLabel("Current stylesheet: <none>", self)
        layout.addWidget(self.file_path_label)

        # Create splitter for raw file and editor
        self.splitter = QSplitter(Qt.Vertical)

        # Raw file view
        self.raw_file_view = QTextEdit(self)
        self.raw_file_view.setReadOnly(True)
        self.raw_file_view.setAcceptRichText(False)
        self.raw_file_view.setPlaceholderText(
            "Raw stylesheet file content will appear here"
        )

        # Editor section with search bar
        self.editor_widget = QWidget(self)
        editor_layout = QVBoxLayout(self.editor_widget)

        self.search_bar = QLineEdit(self)
        self.search_bar.setPlaceholderText("Search in stylesheet (Ctrl+F)")
        self.search_bar.textChanged.connect(self.onSearchTextChanged)

        self.style_text_edit = QTextEdit(self)
        self.style_text_edit.setAcceptRichText(False)
        self.style_text_edit.textChanged.connect(self.onStyleTextChanged)

        editor_layout.addWidget(self.search_bar)
        editor_layout.addWidget(self.style_text_edit)

        # Add to splitter
        self.splitter.addWidget(self.raw_file_view)
        self.splitter.addWidget(self.editor_widget)
        self.splitter.setSizes([200, 400])  # Initial sizes

        layout.addWidget(self.splitter)

        # Add buttons
        button_layout = QHBoxLayout()

        self.load_button = QPushButton("Load File", self)
        self.load_button.clicked.connect(self.onLoadFile)

        self.apply_button = QPushButton("Apply", self)
        self.apply_button.clicked.connect(self.onApplyButton)
        self.apply_button.setEnabled(False)

        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.onSaveFile)

        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.save_button)

        layout.addLayout(button_layout)

        # Shortcuts
        QShortcut(QKeySequence(Qt.Key_F3), self).activated.connect(self.onNextSearchHit)
        QShortcut(QKeySequence(Qt.CTRL + Qt.Key_F), self).activated.connect(
            self.onFocusSearchBar
        )
        QShortcut(QKeySequence(Qt.CTRL + Qt.Key_S), self).activated.connect(
            self.applyStyleSheet
        )
        QShortcut(QKeySequence(Qt.CTRL + Qt.ALT + Qt.Key_Z), self).activated.connect(
            self.onUndo
        )
        QShortcut(QKeySequence(Qt.CTRL + Qt.ALT + Qt.Key_Y), self).activated.connect(
            self.onRedo
        )
        QShortcut(QKeySequence(Qt.Key_F1), self).activated.connect(self.onHelp)

        # Load current stylesheet
        self.loadStyleSheet()

    def onLoadFile(self):
        """Load a stylesheet from a file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Stylesheet", "", "Stylesheet Files (*.qss *.css);;All Files (*)"
        )

        if file_path:
            try:
                with open(file_path, "r") as f:
                    content = f.read()

                self.current_file = file_path
                self.file_path_label.setText(f"Current stylesheet: {file_path}")
                self.raw_file_view.setPlainText(content)
                self.style_text_edit.setPlainText(content)

                # Add to tape
                self.tape = [content]
                self.tape_pos = 0
                self.apply_button.setEnabled(False)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")

    def onSaveFile(self):
        """Save the current stylesheet to a file."""
        if self.current_file:
            default_path = self.current_file
        else:
            default_path = ""

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Stylesheet",
            default_path,
            "Stylesheet Files (*.qss *.css);;All Files (*)",
        )

        if file_path:
            try:
                with open(file_path, "w") as f:
                    f.write(self.style_text_edit.toPlainText())

                self.current_file = file_path
                self.file_path_label.setText(f"Current stylesheet: {file_path}")
                self.raw_file_view.setPlainText(self.style_text_edit.toPlainText())
                QMessageBox.information(
                    self, "Success", "Stylesheet saved successfully."
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")

    def onUndo(self, checked=False):
        if self.tape_pos <= 0:
            return
        self.tape_pos -= 1
        self.style_text_edit.setPlainText(self.tape[self.tape_pos])
        self.applyStyleSheet(stateless=True)

    def onRedo(self, checked=False):
        if self.tape_pos >= len(self.tape) - 1:
            return
        self.tape_pos += 1
        self.style_text_edit.setPlainText(self.tape[self.tape_pos])
        self.applyStyleSheet(stateless=True)

    def onHelp(self):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Help")
        msg_box.setText("Available shortcuts:")
        msg_box.setInformativeText(
            dedent(
                """\
            F1: show help dialog
            Ctrl+S: apply current changes
            Ctrl+F: go to search bar
            F3: go to next search hit
            Ctrl+Alt+Z: revert to last applied style sheet
            Ctrl+Alt+Y: redo last reverted style sheet
        """
            )
        )
        msg_box.exec_()

    def onSearchTextChanged(self, text):
        if not self.style_text_edit.find(text):
            self.search_bar.setStyleSheet("color: red;")
            self.style_text_edit.moveCursor(QTextCursor.Start)
        else:
            self.search_bar.setStyleSheet("color: green;")

    def onNextSearchHit(self):
        search = self.search_bar.text()
        if not self.style_text_edit.find(search):
            self.style_text_edit.moveCursor(QTextCursor.Start)
            self.style_text_edit.find(search)

    def onFocusSearchBar(self):
        self.search_bar.setFocus()

    def onStyleTextChanged(self):
        self.apply_button.setEnabled(True)

    def onApplyButton(self, checked=False):
        self.applyStyleSheet()
        # Update the raw view with the current content
        self.raw_file_view.setPlainText(self.style_text_edit.toPlainText())

    def loadStyleSheet(self):
        ss = QApplication.instance().styleSheet()
        self.tape = [ss]
        self.tape_pos = 0
        self.style_text_edit.setPlainText(ss)
        self.raw_file_view.setPlainText(ss)
        self.apply_button.setEnabled(False)
        self.file_path_label.setText("Current stylesheet: <from application>")

    def applyStyleSheet(self, stateless=False):
        ss = self.style_text_edit.toPlainText()
        QApplication.instance().setStyleSheet(ss)
        if not stateless:
            # Discard any "future" states
            self.tape = self.tape[: self.tape_pos + 1]
            self.tape.append(ss)
            self.tape_pos += 1
        self.apply_button.setEnabled(False)


class ObjectInspector(QWidget):
    """
    Qt widget inspector.
    Automatically inspects widgets on hover without stealing focus.
    Presents inspection data in a tree widget with collapsible categories.
    Includes deep metadata: style engine, cursor, autofill, focus, dynamic properties, palette, actions.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Store reference to main window for accessing the status bar
        self.main_window = self.get_main_window()

        # Always on top as a tool window, but does not grab focus
        self.setWindowFlags(self.windowFlags() | Qt.Tool)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

        # Flag to track if updates are suspended
        self.updates_suspended = False
        # Flag to track if updates are manually suspended (for context menu)
        self.menu_suspended = False

        # Flag for expand on updates
        self.expand_on_updates = False

        # For status message handling
        self.status_timer = QTimer(self)
        self.status_timer.setSingleShot(True)
        self.status_timer.timeout.connect(self.restore_default_status)

        # Tracking categories for tree widget
        self.categories = {}

        # Store context menu data
        self.context_menu_data = None

        # Install event filter for hover events
        app = QApplication.instance()
        app.installEventFilter(self)

        # Shortcut F7 to select parent
        sch = QShortcut(QKeySequence(Qt.Key_F7), self)
        sch.setContext(Qt.ApplicationShortcut)
        sch.activated.connect(self.select_parent)

        self.selected_widget = None

        self._create_ui()
        self._set_tree_style()

        # Enable key tracking
        self.setFocusPolicy(Qt.StrongFocus)

        # Show default status
        self.set_status_message("Hold Ctrl OR Shift to suspend updates")

    def get_main_window(self):
        """Get reference to the main window."""
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, QMainWindow):
                return parent
            parent = parent.parent()
        return None

    def _create_ui(self):
        # Top buttons
        self.btn_select_parent = QPushButton("Select parent", self)
        self.btn_select_parent.released.connect(self.select_parent)

        self.btn_copy = QPushButton("Copy All", self)
        self.btn_copy.released.connect(self.copy_to_clipboard)

        self.btn_export = QPushButton("Export...", self)
        self.btn_export.released.connect(self.export_data)

        self.btn_expand_all = QPushButton("Expand All", self)
        self.btn_expand_all.released.connect(self.expand_all_categories)

        self.btn_collapse_all = QPushButton("Collapse All", self)
        self.btn_collapse_all.released.connect(self.collapse_all_categories)

        self.chk_expand_on_updates = QCheckBox("Expand on Updates", self)
        self.chk_expand_on_updates.stateChanged.connect(self.toggle_expand_on_updates)

        # Help label
        self.lbl_help = QLabel(
            "Hover over any widget to inspect it; press F7 to select its parent. Right-click for context menu.",
            self,
        )

        spacer = QWidget(self)
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # Top controls layout
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(2, 2, 2, 2)
        top_layout.setSpacing(4)
        top_layout.addWidget(self.btn_select_parent)
        top_layout.addWidget(self.btn_copy)
        top_layout.addWidget(self.btn_export)
        top_layout.addWidget(self.btn_expand_all)
        top_layout.addWidget(self.btn_collapse_all)
        top_layout.addWidget(self.chk_expand_on_updates)
        top_layout.addWidget(spacer)

        # Second row layout
        second_row_layout = QHBoxLayout()
        second_row_layout.setContentsMargins(2, 2, 2, 2)
        second_row_layout.setSpacing(4)
        second_row_layout.addWidget(self.lbl_help)
        second_row_layout.addStretch()

        # Inspection tree
        self.tree_inspection = QTreeWidget(self)
        self.tree_inspection.setColumnCount(2)
        self.tree_inspection.setHeaderLabels(["Property", "Value"])
        self.tree_inspection.header().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.tree_inspection.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tree_inspection.setEditTriggers(QTreeWidget.NoEditTriggers)
        self.tree_inspection.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tree_inspection.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_inspection.customContextMenuRequested.connect(self.show_context_menu)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(6)
        main_layout.addLayout(top_layout)
        main_layout.addLayout(second_row_layout)
        main_layout.addWidget(self.tree_inspection)
        self.setLayout(main_layout)

    def set_status_message(self, message, timeout=0):
        """
        Set a status message with optional timeout.
        If timeout > 0, the message will be shown for that many milliseconds
        and then revert to the default message.
        """
        # Cancel any pending timer
        self.status_timer.stop()

        # Set the new message on the main window's status bar if available
        if self.main_window and hasattr(self.main_window, "statusBar"):
            self.main_window.statusBar().showMessage(message)

        # If timeout specified, schedule restoration of default message
        if timeout > 0:
            self.status_timer.start(timeout)

    def restore_default_status(self):
        """Restore the default status message based on the current state."""
        if self.updates_suspended:
            self.set_status_message("Updates suspended")
        else:
            self.set_status_message("Hold Ctrl OR Shift to suspend updates")

    def toggle_expand_on_updates(self, state):
        """Toggle whether categories should be expanded on inspection updates."""
        self.expand_on_updates = state == Qt.Checked

    def expand_all_categories(self):
        """Expand all categories in the tree."""
        for category_idx in range(self.tree_inspection.topLevelItemCount()):
            category = self.tree_inspection.topLevelItem(category_idx)
            category.setExpanded(True)

    def collapse_all_categories(self):
        """Collapse all categories in the tree."""
        for category_idx in range(self.tree_inspection.topLevelItemCount()):
            category = self.tree_inspection.topLevelItem(category_idx)
            category.setExpanded(False)

    def show_context_menu(self, position):
        """Show a context menu for the tree item at the given position."""
        item = self.tree_inspection.itemAt(position)
        if not item:
            return

        # Suspend updates while the context menu is active
        self.menu_suspended = True
        self.updates_suspended = True
        self.set_status_message("Updates suspended for menu operation")

        menu = QMenu(self)
        # Connect to aboutToHide to resume updates if menu is cancelled
        menu.aboutToHide.connect(self.resume_updates_after_menu)

        # Store item data before showing menu
        is_category = item.parent() is None

        if is_category:
            # For category, store all property data
            category_name = item.text(0)
            properties = []

            for prop_idx in range(item.childCount()):
                prop_item = item.child(prop_idx)
                properties.append((prop_item.text(0), prop_item.text(1)))

            self.context_menu_data = {
                "type": "category",
                "name": category_name,
                "properties": properties,
                "item": item,  # Keep reference for expand/collapse actions
            }

            # Category item
            copy_action = menu.addAction("Copy Category")
            copy_action.triggered.connect(self.copy_context_category)

            expand_action = menu.addAction("Expand")
            expand_action.triggered.connect(self.expand_context_category)

            collapse_action = menu.addAction("Collapse")
            collapse_action.triggered.connect(self.collapse_context_category)
        else:
            # For property, just store the property data
            prop_name = item.text(0)
            prop_value = item.text(1)

            self.context_menu_data = {
                "type": "property",
                "name": prop_name,
                "value": prop_value,
            }

            # Property item
            copy_action = menu.addAction("Copy Property")
            copy_action.triggered.connect(self.copy_context_property)

        menu.exec_(self.tree_inspection.viewport().mapToGlobal(position))

    def resume_updates_after_menu(self):
        """Resume updates after the context menu is hidden."""
        # Only resume if suspension was due to menu (not Ctrl/Shift keys)
        if self.menu_suspended:
            self.menu_suspended = False
            # Check if Ctrl/Shift are still pressed
            modifiers = QApplication.keyboardModifiers()
            if not (modifiers & Qt.ControlModifier or modifiers & Qt.ShiftModifier):
                self.updates_suspended = False
                self.restore_default_status()

    def copy_context_category(self):
        """Copy all properties in the context menu category to the clipboard."""
        if not self.context_menu_data or self.context_menu_data["type"] != "category":
            return

        category_name = self.context_menu_data["name"]
        properties = self.context_menu_data["properties"]

        lines = [f"{category_name}:"]
        for prop_name, prop_value in properties:
            lines.append(f"  {prop_name}: {prop_value}")

        QApplication.instance().clipboard().setText("\n".join(lines))
        self.set_status_message(f"Category '{category_name}' copied to clipboard", 3000)

        # Resume updates after copy
        self.resume_updates_after_menu()

    def copy_context_property(self):
        """Copy a single property from the context menu to the clipboard."""
        if not self.context_menu_data or self.context_menu_data["type"] != "property":
            return

        prop_name = self.context_menu_data["name"]
        prop_value = self.context_menu_data["value"]

        text = f"{prop_name}: {prop_value}"
        QApplication.instance().clipboard().setText(text)
        self.set_status_message(f"Property '{prop_name}' copied to clipboard", 3000)

        # Resume updates after copy
        self.resume_updates_after_menu()

    def expand_context_category(self):
        """Expand the category from the context menu."""
        if not self.context_menu_data or self.context_menu_data["type"] != "category":
            return

        item = self.context_menu_data["item"]
        if item and self.tree_inspection.indexOfTopLevelItem(item) >= 0:
            item.setExpanded(True)

        # Resume updates after expand
        self.resume_updates_after_menu()

    def collapse_context_category(self):
        """Collapse the category from the context menu."""
        if not self.context_menu_data or self.context_menu_data["type"] != "category":
            return

        item = self.context_menu_data["item"]
        if item and self.tree_inspection.indexOfTopLevelItem(item) >= 0:
            item.setExpanded(False)

        # Resume updates after collapse
        self.resume_updates_after_menu()

    def export_data(self):
        """Export the inspection data to a file."""
        if not self.selected_widget:
            self.set_status_message("No widget selected to export", 3000)
            return

        # Suspend updates during export
        was_suspended = self.updates_suspended
        self.updates_suspended = True
        self.set_status_message("Updates suspended for export operation")

        # Prepare data for export
        data = self._get_data_as_dict()

        # Ask user for file location and format
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Inspection Data",
            f"widget_inspection_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "JSON Files (*.json);;INI Files (*.ini);;All Files (*)",
        )

        if not file_path:
            # Resume updates if export was cancelled
            if not was_suspended:
                self.updates_suspended = False
                self.restore_default_status()
            return  # User canceled

        try:
            if file_path.endswith(".json") or "JSON" in selected_filter:
                self._export_as_json(file_path, data)
            elif file_path.endswith(".ini") or "INI" in selected_filter:
                self._export_as_ini(file_path, data)
            else:
                # Default to JSON if no extension specified
                if not "." in file_path:
                    file_path += ".json"
                self._export_as_json(file_path, data)

            self.set_status_message(f"Exported data to {file_path}", 3000)
        except Exception as e:
            self.set_status_message(f"Error exporting data: {str(e)}", 5000)
        finally:
            # Resume updates if they weren't suspended before
            if not was_suspended:
                self.updates_suspended = False
                # Status will be restored by the timer set in set_status_message

    def _get_data_as_dict(self):
        """Get the inspection data as a nested dictionary."""
        data = {}

        # Iterate through all categories
        for category_idx in range(self.tree_inspection.topLevelItemCount()):
            category = self.tree_inspection.topLevelItem(category_idx)
            category_name = category.text(0)
            category_data = {}

            # Iterate through properties in this category
            for prop_idx in range(category.childCount()):
                prop_item = category.child(prop_idx)
                prop_name = prop_item.text(0)
                prop_value = prop_item.text(1)
                category_data[prop_name] = prop_value

            data[category_name] = category_data

        return data

    def _export_as_json(self, file_path, data):
        """Export data as JSON."""
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    def _export_as_ini(self, file_path, data):
        """Export data as INI file."""
        config = configparser.ConfigParser()

        for category, props in data.items():
            # Replace any characters not allowed in INI section names
            safe_category = category.replace("[", "(").replace("]", ")")
            config[safe_category] = props

        with open(file_path, "w") as f:
            config.write(f)

    def _set_tree_style(self):
        font = QFont("Monospace")
        font.setStyleHint(QFont.TypeWriter)
        self.tree_inspection.setFont(font)

    def _update_suspended_state(self):
        """Check and update the suspended state based on modifier keys."""
        # Don't change state if menu is active
        if self.menu_suspended:
            return True

        modifiers = QApplication.keyboardModifiers()
        was_suspended = self.updates_suspended
        self.updates_suspended = bool(
            modifiers & Qt.ControlModifier or modifiers & Qt.ShiftModifier
        )

        if self.updates_suspended != was_suspended:
            if self.updates_suspended:
                self.set_status_message("Updates suspended")
            else:
                self.set_status_message("Hold Ctrl OR Shift to suspend updates")

        return self.updates_suspended

    def eventFilter(self, obj, event):
        # Check if modifiers changed on any event
        if event.type() in (QEvent.KeyPress, QEvent.KeyRelease):
            self._update_suspended_state()

        # Handle Enter events for widget inspection
        elif event.type() == QEvent.Enter and isinstance(obj, QWidget):
            if obj is self or self.isAncestorOf(obj):
                return super().eventFilter(obj, event)

            # Check keyboard modifiers
            if not self._update_suspended_state() and obj is not self:
                self._inspect_widget(obj)

        return super().eventFilter(obj, event)

    def select_parent(self):
        if not self.selected_widget:
            return
        parent = self.selected_widget.parent()
        if parent and parent.inherits("QWidget"):
            self._inspect_widget(parent)

    def _get_specific_css_selector(self, widget):
        """Generate the most specific CSS selector for a widget."""
        widget_type = widget.metaObject().className()
        object_name = widget.objectName()

        if object_name:
            return f'{widget_type}[objectName="{object_name}"]'
        else:
            return widget_type

    def _get_css_selector_with_parent(self, widget):
        """Generate a CSS selector that includes the immediate parent."""
        if not widget.parent():
            return self._get_specific_css_selector(widget)

        parent_selector = self._get_specific_css_selector(widget.parent())
        widget_selector = self._get_specific_css_selector(widget)
        return f"{parent_selector} > {widget_selector}"

    def _get_full_hierarchy_css_selector(self, widget):
        """Generate a CSS selector with the full widget hierarchy from root to widget."""
        selectors = []
        current = widget

        while current:
            selectors.insert(0, self._get_specific_css_selector(current))
            current = current.parent()

        return " > ".join(selectors)

    def _add_css_selector_hierarchy(self, widget, css_selectors):
        """Add CSS selectors for the widget and all its parents to the css_selectors dict."""
        # Add the most specific selector for this widget
        css_selectors["CSS Selector (Specific)"] = self._get_specific_css_selector(
            widget
        )

        # Build a list of all ancestors
        ancestors = []
        current = widget
        while current.parent():
            ancestors.insert(0, current.parent())
            current = current.parent()

        # Add a selector for each level of the hierarchy
        if ancestors:
            # First, add the immediate parent relationship
            css_selectors["CSS Selector (With Parent)"] = (
                self._get_css_selector_with_parent(widget)
            )

            # Then add the full hierarchy selector
            css_selectors["CSS Selector (Full Hierarchy)"] = (
                self._get_full_hierarchy_css_selector(widget)
            )

            # Add intermediate levels if there are more than one parent
            if len(ancestors) > 1:
                # Start building from the root
                path_selectors = []
                for i, ancestor in enumerate(ancestors):
                    path_selectors.append(self._get_specific_css_selector(ancestor))
                    # Skip the immediate parent (already covered) and full hierarchy (already covered)
                    if 0 < i < len(ancestors) - 1:
                        path_selectors_copy = path_selectors.copy()
                        path_selectors_copy.append(
                            self._get_specific_css_selector(widget)
                        )
                        css_selectors[f"CSS Selector (With {i+1} Parents)"] = (
                            " > ".join(path_selectors_copy)
                        )

    def _add_category(self, name, collapsed=True):
        """Add a category to the tree widget."""
        category = QTreeWidgetItem(self.tree_inspection)
        category.setText(0, name)
        category.setExpanded(not collapsed and not self.expand_on_updates)
        self.categories[name] = category
        return category

    def _add_property(self, category, prop, val, icon=None):
        """Add a property to a category in the tree widget."""
        item = QTreeWidgetItem(category)
        item.setText(0, prop)
        item.setText(1, val)
        if icon:
            item.setIcon(1, icon)
        return item

    def _inspect_widget(self, widget):
        # Disconnect previous destroyed signal
        if self.selected_widget is not None:
            try:
                self.selected_widget.destroyed.disconnect(self.on_widget_destroyed)
            except TypeError:
                pass

        self.selected_widget = widget

        # Clear existing tree and categories
        self.tree_inspection.clear()
        self.categories = {}

        # Clear context menu data
        self.context_menu_data = None

        if not widget:
            self._add_property(self.tree_inspection, "<no widget>", "")
            return

        # Create categories with appropriate collapsed state (might be overridden by expand_on_updates)
        self._add_category("Screen Information", collapsed=True)
        self._add_category("Basic Information", collapsed=False)
        self._add_category("CSS Selectors", collapsed=True)
        self._add_category("Parent Information", collapsed=False)
        self._add_category("Widget State", collapsed=False)
        self._add_category("Style & Appearance", collapsed=False)
        self._add_category("Layout", collapsed=False)
        self._add_category("Properties", collapsed=False)
        self._add_category("Palette Colors", collapsed=True)
        self._add_category("Interactivity", collapsed=False)
        self._add_category("Miscellaneous", collapsed=False)

        # Screen info
        screen = QApplication.instance().primaryScreen()
        res = screen.size()
        phy = screen.physicalSize()
        dpi = screen.logicalDotsPerInch()
        self._add_property(
            self.categories["Screen Information"],
            "Screen Resolution",
            f"{res.width()}×{res.height()} px",
        )
        self._add_property(
            self.categories["Screen Information"],
            "Physical Size",
            f"{phy.width()}×{phy.height()} mm",
        )
        self._add_property(self.categories["Screen Information"], "DPI", f"{dpi:.1f}")

        # Basic info
        self._add_property(
            self.categories["Basic Information"],
            "Type",
            widget.metaObject().className(),
        )
        self._add_property(
            self.categories["Basic Information"],
            "Name",
            widget.objectName() or "<none>",
        )
        self._add_property(
            self.categories["Basic Information"],
            "Children Count",
            str(len(widget.children())),
        )

        # CSS selectors
        css_selectors = {}
        self._add_css_selector_hierarchy(widget, css_selectors)
        for name, value in css_selectors.items():
            self._add_property(self.categories["CSS Selectors"], name, value)

        # Parent info
        parent = widget.parent()
        self._add_property(
            self.categories["Parent Information"],
            "Parent Type",
            parent.metaObject().className() if parent else "<none>",
        )
        self._add_property(
            self.categories["Parent Information"],
            "Parent Name",
            parent.objectName() if parent and parent.objectName() else "<none>",
        )

        # Widget state
        self._add_property(
            self.categories["Widget State"], "Enabled", str(widget.isEnabled())
        )
        self._add_property(
            self.categories["Widget State"], "Visible", str(widget.isVisible())
        )
        self._add_property(
            self.categories["Widget State"], "Under Mouse", str(widget.underMouse())
        )
        self._add_property(
            self.categories["Widget State"], "Has Focus", str(widget.hasFocus())
        )

        # Style & Appearance
        self._add_property(
            self.categories["Style & Appearance"],
            "Style Engine",
            widget.style().metaObject().className(),
        )
        self._add_property(
            self.categories["Style & Appearance"],
            "Cursor Shape",
            str(widget.cursor().shape()),
        )
        self._add_property(
            self.categories["Style & Appearance"],
            "AutoFillBackground",
            str(widget.autoFillBackground()),
        )
        self._add_property(
            self.categories["Style & Appearance"],
            "Focus Policy",
            str(widget.focusPolicy()),
        )

        # Font
        font = widget.font()
        self._add_property(
            self.categories["Style & Appearance"],
            "Font",
            f"family={font.family()}, size={font.pointSize()}, weight={font.weight()}, italic={font.italic()}",
        )

        # Layout
        geom = widget.geometry()
        self._add_property(
            self.categories["Layout"],
            "Geometry",
            f"x={geom.x()}, y={geom.y()}, width={geom.width()}, height={geom.height()}",
        )

        # Layout margins
        layout = widget.layout()
        if layout:
            m = layout.contentsMargins()
            self._add_property(
                self.categories["Layout"],
                "Layout Margins",
                f"left={m.left()}, top={m.top()}, right={m.right()}, bottom={m.bottom()}",
            )
        else:
            self._add_property(
                self.categories["Layout"], "Layout Margins", "<no layout>"
            )

        # Style sheet
        self._add_property(
            self.categories["Layout"], "StyleSheet", widget.styleSheet() or "<none>"
        )

        # Dynamic properties
        dp = [name.data().decode() for name in widget.dynamicPropertyNames()]
        self._add_property(
            self.categories["Properties"],
            "Dynamic Properties",
            ", ".join(dp) or "<none>",
        )

        # Palette colors - dynamic version
        pal = widget.palette()
        roles = {}
        for i in range(QPalette.NColorRoles):
            try:
                col = pal.color(i)
                if not col.isValid():
                    continue

                # find the matching attribute name on QPalette
                name = next(
                    (
                        name
                        for name, val in QPalette.__dict__.items()
                        if isinstance(val, int) and val == i
                    ),
                    f"Role_{i}",  # Fallback name if not found
                )
                # store the integer role (not the color string)
                roles[name] = i
            except Exception:
                continue  # Skip any role that causes an error

        # now pull each color back out and append
        for name in sorted(roles):
            role = roles[name]
            color = pal.color(role).name()
            pix = QPixmap(16, 16)
            pix.fill(QColor(color))
            icon = QIcon(pix)
            self._add_property(self.categories["Palette Colors"], name, color, icon)

        # Shortcuts
        shortcuts = widget.findChildren(QShortcut)
        sc_list = [sc.key().toString() for sc in shortcuts]
        self._add_property(
            self.categories["Interactivity"],
            "Shortcuts",
            ", ".join(sc_list) or "<none>",
        )

        # Actions
        actions = widget.actions()
        act_list = [f"{act.text()} ({act.shortcut().toString()})" for act in actions]
        self._add_property(
            self.categories["Interactivity"], "Actions", "; ".join(act_list) or "<none>"
        )

        # Paint engine
        engine = widget.paintEngine()
        self._add_property(
            self.categories["Miscellaneous"],
            "Paint Engine",
            engine.__class__.__name__ if engine else "<none>",
        )

        # If expand_on_updates is enabled, expand all categories
        if self.expand_on_updates:
            self.expand_all_categories()

        # Connect destroyed signal
        widget.destroyed.connect(self.on_widget_destroyed)

    def copy_to_clipboard(self):
        if not self.selected_widget:
            return

        # Suspend updates during copy
        was_suspended = self.updates_suspended
        self.updates_suspended = True
        self.set_status_message("Updates suspended for copy operation")

        lines = []

        # Iterate through all categories
        for category_idx in range(self.tree_inspection.topLevelItemCount()):
            category = self.tree_inspection.topLevelItem(category_idx)
            category_name = category.text(0)
            lines.append(f"{category_name}:")

            # Iterate through properties in this category
            for prop_idx in range(category.childCount()):
                prop_item = category.child(prop_idx)
                prop_name = prop_item.text(0)
                prop_value = prop_item.text(1)
                lines.append(f"  {prop_name}: {prop_value}")

        QApplication.instance().clipboard().setText("\n".join(lines))
        self.set_status_message("All data copied to clipboard", 3000)

        # Resume updates if they weren't suspended before
        if not was_suspended:
            self.updates_suspended = False
            # Status will be restored by timer set in set_status_message

    def on_widget_destroyed(self, _):
        self.selected_widget = None


class QtStyleSelector(QWidget):
    """Widget for selecting Qt style and toggling dark mode."""

    def __init__(self, parent=None):
        super(QtStyleSelector, self).__init__(parent)
        self.setup_ui()
        self.load_saved_settings()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Style selector group
        style_group = QWidget(self)
        style_layout = QHBoxLayout(style_group)

        style_label = QLabel("Qt Style:", self)
        self.style_combo = QComboBox(self)
        self.style_combo.addItems(QStyleFactory.keys())

        style_layout.addWidget(style_label)
        style_layout.addWidget(self.style_combo)

        # Dark mode toggle
        self.dark_mode_checkbox = QCheckBox("Dark Mode", self)

        # Buttons
        button_layout = QHBoxLayout()
        self.apply_button = QPushButton("Apply", self)
        self.apply_button.clicked.connect(self.apply_settings)

        self.save_button = QPushButton("Save Settings", self)
        self.save_button.clicked.connect(self.save_settings)

        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.save_button)

        # Add to main layout
        layout.addWidget(style_group)
        layout.addWidget(self.dark_mode_checkbox)
        layout.addLayout(button_layout)
        layout.addStretch()

    def load_saved_settings(self):
        try:
            style, dark_mode = load_style()

            # Set style combo box
            index = self.style_combo.findText(style)
            if index >= 0:
                self.style_combo.setCurrentIndex(index)

            # Set dark mode checkbox
            self.dark_mode_checkbox.setChecked(dark_mode)

        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Failed to load settings: {str(e)}")

    def apply_settings(self):
        style = self.style_combo.currentText()
        dark_mode = self.dark_mode_checkbox.isChecked()

        # Apply style
        QApplication.instance().setStyle(QStyleFactory.create(style))

        # Apply dark mode
        toggle_dark_theme(dark_mode)

        QMessageBox.information(self, "Success", "Settings applied successfully.")

    def save_settings(self):
        try:
            style = self.style_combo.currentText()
            dark_mode = self.dark_mode_checkbox.isChecked()

            save_style(style, dark_mode)
            QMessageBox.information(self, "Success", "Settings saved successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings: {str(e)}")


class IDAThemingToolkit(QMainWindow):
    """
    Main application window for the IDA Theming Toolkit.
    Provides access to widget inspector, style sheet editor, and theme settings.
    """

    def __init__(self, parent=None):
        super(IDAThemingToolkit, self).__init__(parent)
        self.setWindowTitle("IDA Theming Toolkit")
        self.resize(1200, 800)

        # Create central widget with tabs
        self.central_widget = QTabWidget(self)
        self.setCentralWidget(self.central_widget)

        # Create status bar first so the object inspector can access it
        self.statusBar().showMessage("Ready")

        # Create tabs
        self.object_inspector = ObjectInspector(
            self
        )  # Pass self as parent so it can access the statusBar
        self.stylesheet_editor = StyleSheetWidget(self)
        self.style_selector = QtStyleSelector(self)

        # Add tabs
        self.central_widget.addTab(self.object_inspector, "Widget Inspector")
        self.central_widget.addTab(self.stylesheet_editor, "Stylesheet Editor")
        self.central_widget.addTab(self.style_selector, "Qt Style & Theme")

        # Set up tab change handler to update status bar
        self.central_widget.currentChanged.connect(self.on_tab_changed)

        # Load initial settings
        self.load_initial_settings()

    def on_tab_changed(self, index):
        """Update the status bar when tabs are changed."""
        if index == 0:  # Widget Inspector tab
            self.statusBar().showMessage("Hold Ctrl OR Shift to suspend updates")
        elif index == 1:  # Stylesheet Editor tab
            self.statusBar().showMessage("Ready - Stylesheet Editor")
        elif index == 2:  # Qt Style & Theme tab
            self.statusBar().showMessage("Ready - Style & Theme Settings")

    @classmethod
    def load_initial_settings(cls):
        try:
            style, dark_mode = load_style()

            # Apply saved settings if they exist
            if style:
                QApplication.instance().setStyle(QStyleFactory.create(style))

            if dark_mode:
                toggle_dark_theme(dark_mode)

            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
            QApplication.setAttribute(
                Qt.HighDpiScaleFactorRoundingPolicy.PassThrough, True
            )
            idaapi.msg(
                f"{PLUGIN_NAME}: Applied style: {style}, Dark mode: {dark_mode}{os.linesep}"
            )
        except Exception as e:
            if cls is IDAThemingToolkit:
                idaapi.msg("Failed to load settings: {str(e)}\n")
            else:
                cls.statusBar().showMessage(f"Failed to load settings: {str(e)}")


class IDARootPlugin(idaapi.plugin_t):
    flags = idaapi.PLUGIN_FIX
    comment = "IDA Theming Toolkit"
    help = "Tools for customizing IDA's appearance"
    wanted_name = "IDA Theming Toolkit"
    wanted_hotkey = "Alt-T"

    def init(self):
        IDAThemingToolkit.load_initial_settings()
        return idaapi.PLUGIN_KEEP

    def run(self, arg):
        # Launch the toolkit
        toolkit = IDAThemingToolkit()
        toolkit.show()

    def term(self):
        pass


# Entry point for IDA Plugin
def PLUGIN_ENTRY():
    return IDARootPlugin()


if __name__ == "__main__":
    toolkit = IDAThemingToolkit()
    toolkit.show()
