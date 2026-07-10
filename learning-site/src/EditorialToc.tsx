import { useEffect, useState } from "react";
import { createPortal } from "react-dom";

export type EditorialTocItem = {
  id: string;
  title: string;
  level?: 2 | 3;
};

export default function EditorialToc({ items, accent = "#7b2e2b" }: { items: EditorialTocItem[]; accent?: string }) {
  const [activeId, setActiveId] = useState(items[0]?.id ?? "");
  const [visible, setVisible] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [darkSurface, setDarkSurface] = useState(false);
  const activeItem = items.find(item => item.id === activeId) ?? items[0];

  useEffect(() => {
    const sections = items.map(item => document.getElementById(item.id)).filter(Boolean) as HTMLElement[];
    const updateFromScroll = () => {
      const first = sections[0];
      const last = sections[sections.length - 1];
      if (!first || !last) return;
      setVisible(window.scrollY >= first.offsetTop - 180 && window.scrollY <= last.offsetTop + last.offsetHeight - 120);
      const probe = window.scrollY + Math.min(260, window.innerHeight * .34);
      const current = sections.filter(section => section.offsetTop <= probe).at(-1) ?? first;
      setActiveId(current.id);
      const color = getComputedStyle(current).backgroundColor;
      const channels = color.match(/[\d.]+/g)?.map(Number) ?? [];
      const [red = 245, green = 242, blue = 234, alpha = 1] = channels;
      setDarkSurface(alpha !== 0 && red * .299 + green * .587 + blue * .114 < 105);
    };
    updateFromScroll();
    window.addEventListener("scroll", updateFromScroll, { passive: true });
    window.addEventListener("resize", updateFromScroll);
    return () => {
      window.removeEventListener("scroll", updateFromScroll);
      window.removeEventListener("resize", updateFromScroll);
    };
  }, [items]);

  const navigate = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
    setActiveId(id);
    setMobileOpen(false);
  };

  return createPortal(<aside className={`editorial-side-toc ${visible ? "visible" : ""} ${mobileOpen ? "mobile-open" : ""} ${darkSurface ? "on-dark" : ""}`} aria-label="Table of contents" style={{ "--toc-accent": accent } as React.CSSProperties}>
    <div className="editorial-side-toc-inner">
      <div className="editorial-side-toc-kicker">Contents</div>
      <nav>{items.map(item => <button key={item.id} data-level={item.level ?? 2} data-active={activeId === item.id} onClick={() => navigate(item.id)}>{item.title}</button>)}</nav>
    </div>
    <button className="editorial-toc-mobile-trigger" onClick={() => setMobileOpen(value => !value)} aria-expanded={mobileOpen}>
      <span>Contents</span><strong>{activeItem?.title}</strong><i>{mobileOpen ? "×" : "↑"}</i>
    </button>
  </aside>, document.body);
}
